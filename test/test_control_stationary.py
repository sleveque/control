from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, LinearSolver,
    RectangleMesh, SpatialCoordinate, TestFunction,
    TrialFunction, UnitSquareMesh, VectorFunctionSpace, as_vector, assemble,
    cos, div, dot, ds, dx, exp, grad, inner, pi, sin, solve
)
from firedrake.adjoint import (
    Control as Control_ad, ReducedFunctional, compute_derivative,
    continue_annotation, get_working_tape, minimize, pause_annotation,
    set_working_tape
)

from control.control import Stationary
from control.preconditioner import ConstantNullspace

from functools import partial

import numpy as np
import ufl
import pytest


@pytest.fixture(autouse=True, scope="module")
def cleanup():
    with set_working_tape():
        yield


def test_stationary_linear_control():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 2)

    test_0 = TestFunction(space_0)

    def forw_diff_operator(trial, test, u):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx + inner(trial, test) * dx

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(
            cos(pi * X[0] - 0.5 * pi) * cos(pi * X[1] - 0.5 * pi) + 1.)

        # true v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0]
        y = X[1]

        # force function
        f = Function(space)

        f.interpolate(
            2.0 * pi**2 * cos(pi * x - 0.5 * pi) * cos(pi * y - 0.5 * pi))

        return inner(f, test) * dx

    beta = 1.0e-3

    my_control_stationary = Stationary(
        space_0, forw_diff_operator, desired_state=desired_state,
        force_function=force_f, beta=beta, bcs_v=None)

    v_ref = Function(space_0)
    zeta_ref = Function(space_0)
    v_ref.interpolate(X[0] * exp(X[1]))
    zeta_ref.interpolate(sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    v_0 = Function(space_0)
    zeta_0 = Function(space_0)

    b_0 = assemble(
        inner(v_ref, test_0) * dx
        + inner(grad(zeta_ref), grad(test_0)) * dx
        + inner(zeta_ref, test_0) * dx)
    b_1 = assemble(
        inner(grad(v_ref), grad(test_0)) * dx
        + inner(v_ref, test_0) * dx
        - 1.0 / beta * inner(zeta_ref, test_0) * dx)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-14,
                         "absolute_tolerance": 1.0e-14,
                         "monitor_convergence": False}

    # employing Chebyshev for the (1,1)-block
    e_min = 0.25
    e_max = 1.5625

    sp_11block = {"ksp_type": "chebyshev",
                  "pc_type": "jacobi",
                  "ksp_chebyshev_eigenvalues": f"{e_min:.16e}, {e_max:.16e}",
                  "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                  "ksp_chebyshev_esteig_steps": 0,
                  "ksp_chebyshev_esteig_noisy": False,
                  "ksp_max_it": 20,
                  "ksp_atol": 0.0,
                  "ksp_rtol": 0.0}

    auxiliary_sp = {"sp_11block": sp_11block}

    my_control_stationary.linear_solve(
        solver_parameters=solver_parameters,
        auxiliary_sp=auxiliary_sp,
        v_d=b_0, f=b_1, print_error=False,
        outputs=False, plots=False)

    v_0.assign(my_control_stationary._v)
    zeta_0.assign(my_control_stationary._zeta)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-13
    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-13


@pytest.mark.parametrize("degree", tuple(range(1, 2)))
def test_MMS_stationary_linear_Poisson_control(degree):
    p_range = (3, 5)
    beta = 1.0e-3

    def ref_sol_v(x, y):
        return sin(pi * x) * sin(pi * y) * exp(x + y)

    def ref_sol_zeta(x, y):
        return sin(2.0 * pi * x) * sin(2.0 * pi * y)

    def forw_diff_operator(trial, test, v):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)
        zeta = ref_sol_zeta(*X)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(-div(grad(zeta)) + v)

        # true v
        true_v = Function(space, name="true_v")
        true_v.interpolate(v)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)
        zeta = ref_sol_zeta(*X)

        # force function
        f = Function(space)
        f.interpolate(-div(grad(v)) - 1.0 / beta * zeta)

        return inner(f, test) * dx

    v_error_norms = []
    zeta_error_norms = []
    for p in range(*p_range):
        N = 2 ** p
        mesh = UnitSquareMesh(N, N)
        X = SpatialCoordinate(mesh)

        space_0 = FunctionSpace(mesh, "Lagrange", degree)
        bc = DirichletBC(space_0, 0.0, "on_boundary")

        my_control_stationary = Stationary(
            space_0, forw_diff_operator, desired_state=desired_state,
            force_function=force_f, beta=beta, bcs_v=bc)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 500,
                             "relative_tolerance": 1.0e-6,
                             "absolute_tolerance": 1.0e-6,
                             "monitor_convergence": False}

        my_control_stationary.linear_solve(
            solver_parameters=solver_parameters,
            print_error=False, outputs=False, plots=False)

        my_v = Function(space_0, name="my_v")
        my_zeta = Function(space_0, name="my_zeta")

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)

        v_ref = Function(FunctionSpace(mesh, "Lagrange", degree + 2),
                         name="v_ref")
        v_ref.interpolate(ref_sol_v(*X))

        v_error_norm = np.sqrt(abs(assemble(inner(my_v - v_ref,
                                                  my_v - v_ref) * dx)))
        print(f"{degree=} {p=} {N=} {v_error_norm=}")
        v_error_norms.append(v_error_norm)

        zeta_ref = Function(FunctionSpace(mesh, "Lagrange", degree + 2),
                            name="zeta_ref")
        zeta_ref.interpolate(ref_sol_zeta(*X))

        zeta_error_norm = np.sqrt(abs(assemble(
            inner(my_zeta - zeta_ref,
                  my_zeta - zeta_ref) * dx)))
        print(f"{degree=} {p=} {N=} {zeta_error_norm=}")
        zeta_error_norms.append(zeta_error_norm)

    v_error_norms = np.array(v_error_norms)
    v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {v_orders=}")
    min_v_ord = min(v_orders)
    assert min_v_ord > 1.5

    zeta_error_norms = np.array(zeta_error_norms)
    zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {zeta_orders=}")
    min_z_ord = min(zeta_orders)
    assert min_z_ord > 1.5


def test_stationary_incompressible_linear_control():
    mesh_size = 2
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
    X = SpatialCoordinate(mesh)
    space_0 = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)

    test_0 = TestFunction(space_0)
    test_1 = TestFunction(space_1)

    def forw_diff_operator(trial, test, u):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx + inner(trial, test) * dx

    def desired_state(test):
        space = test.function_space()

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(as_vector((0.0, 0.0)))

        # true v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(as_vector((0.0, 0.0)))

        return inner(f, test) * dx

    beta = 1.0e-3

    my_control_stationary = Stationary(
        space_0, forw_diff_operator, desired_state=desired_state,
        force_function=force_f, beta=beta, space_p=space_1,
        bcs_v=DirichletBC(space_0, 0.0, "on_boundary"))

    v_ref = Function(space_0)
    zeta_ref = Function(space_0)
    v_ref.interpolate(
        as_vector([X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]),
                  sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1])]))
    zeta_ref.interpolate(
        as_vector([sin(pi * X[0]) * sin(2.0 * pi * X[1]),
                  sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1])]))

    p_ref = Function(space_1)
    mu_ref = Function(space_1)
    p_ref.interpolate(sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    mu_ref.interpolate(X[0] * exp(X[1]))

    v_0 = Function(space_0)
    zeta_0 = Function(space_0)
    p_0 = Function(space_1)
    mu_0 = Function(space_1)

    b_0 = assemble(inner(v_ref, test_0) * dx
                   + inner(grad(zeta_ref), grad(test_0)) * dx
                   + inner(zeta_ref, test_0) * dx
                   - inner(mu_ref, div(test_0)) * dx)
    b_1 = assemble(inner(grad(v_ref), grad(test_0)) * dx
                   + inner(v_ref, test_0) * dx
                   - 1.0 / beta * inner(zeta_ref, test_0) * dx
                   - inner(p_ref, div(test_0)) * dx)
    b_2 = assemble(- inner(div(v_ref), test_1) * dx)
    b_3 = assemble(- inner(div(zeta_ref), test_1) * dx)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-15,
                         "absolute_tolerance": 1.0e-15,
                         "monitor_convergence": False}

    # employing Chebyshev for the (1,1)-block
    e_min_v = 0.25
    e_max_v = 1.5625
    sp_11block = {
        "ksp_type": "chebyshev",
        "pc_type": "jacobi",
        "ksp_chebyshev_eigenvalues": f"{e_min_v:.16e}, {e_max_v:.16e}",
        "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
        "ksp_chebyshev_esteig_steps": 0,
        "ksp_chebyshev_esteig_noisy": False,
        "ksp_max_it": 20,
        "ksp_atol": 0.0,
        "ksp_rtol": 0.0}

    # employing Chebyshev for the pressure-mass matrix
    e_min_p = 0.25
    e_max_p = 2.25
    sp_M_p = {
        "ksp_type": "chebyshev",
        "pc_type": "jacobi",
        "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",
        "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
        "ksp_chebyshev_esteig_steps": 0,
        "ksp_chebyshev_esteig_noisy": False,
        "ksp_max_it": 20,
        "ksp_atol": 0.0,
        "ksp_rtol": 0.0}

    auxiliary_sp = {"sp_11block": sp_11block,
                    "sp_M_p": sp_M_p}

    my_control_stationary.incompressible_linear_solve(
        ConstantNullspace(), solver_parameters=solver_parameters,
        auxiliary_sp=auxiliary_sp,
        v_d=b_0, f=b_1, div_v=b_2, div_zeta=b_3,
        print_error=False, outputs=False,
        plots=False)

    v_0.assign(my_control_stationary._v)
    zeta_0.assign(my_control_stationary._zeta)
    p_0.assign(my_control_stationary._p)
    mu_0.assign(my_control_stationary._mu)

    mean = assemble(mu_0 * dx)
    with mu_0.dat.vec as b_p:
        b_p.shift(-mean)
    mean = assemble(mu_ref * dx)
    with mu_ref.dat.vec as b_p:
        b_p.shift(-mean)

    mean = assemble(p_0 * dx)
    with p_0.dat.vec as b_p:
        b_p.shift(-mean)
    mean = assemble(p_ref * dx)
    with p_ref.dat.vec as b_p:
        b_p.shift(-mean)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-12
    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-12

    p_error_norm = np.sqrt(abs(assemble(inner(p_0 - p_ref,
                                              p_0 - p_ref) * dx)))
    assert p_error_norm < 1.0e-10
    mu_error_norm = np.sqrt(abs(assemble(inner(mu_0 - mu_ref,
                                               mu_0 - mu_ref) * dx)))
    assert mu_error_norm < 1.0e-10


@pytest.mark.parametrize("degree", tuple(range(2, 3)))
def test_MMS_stationary_Stokes_control(degree):
    p_range = (2, 4)
    beta = 1.0e-3

    def ref_sol_v(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        v = as_vector([X_1 * (X_2 ** 3), (1. / 4.) * (X_1 ** 4 - X_2 ** 4)])

        return v

    def ref_sol_p(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        p = 3. * X_1 ** 2 * X_2 - X_2 ** 3

        return p

    def ref_sol_zeta(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        zeta = as_vector([
            beta * 2. * X_2 * (X_1 ** 2 - 1.) ** 2 * (X_2 ** 2 - 1.),
            - beta * 2. * X_1 * (X_1 ** 2 - 1.) * (X_2 ** 2 - 1.) ** 2])

        return zeta

    def ref_sol_mu(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        mu = beta * 4. * X_1 * X_2

        return mu

    def forw_diff_operator(trial, test, v):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)
        zeta = ref_sol_zeta(*X)
        mu = ref_sol_mu(*X)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(-div(grad(zeta)) + grad(mu) + v)

        # true v
        true_v = Function(space, name="true_v")
        true_v.interpolate(v)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)
        p = ref_sol_p(*X)
        zeta = ref_sol_zeta(*X)

        # force function
        f = Function(space)
        f.interpolate(-div(grad(v)) + grad(p) - 1.0 / beta * zeta)

        return inner(f, test) * dx

    v_error_norms = []
    zeta_error_norms = []
    p_error_norms = []
    mu_error_norms = []
    for p in range(*p_range):
        N = 2 ** p
        mesh = RectangleMesh(N, N, 2.0, 2.0)
        X = SpatialCoordinate(mesh)
        X_1 = X[0] - 1.0
        X_2 = X[1] - 1.0

        space_0 = VectorFunctionSpace(mesh, "Lagrange", degree)
        space_1 = FunctionSpace(mesh, "Lagrange", degree - 1)
        bc = DirichletBC(
            space_0,
            as_vector([
                X_1 * (X_2 ** 3),
                (1. / 4.) * (X_1 ** 4 - X_2 ** 4)]),
            "on_boundary")

        my_control_stationary = Stationary(
            space_0, forw_diff_operator, desired_state=desired_state,
            force_function=force_f, beta=beta, space_p=space_1, bcs_v=bc)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 200,
                             "relative_tolerance": 1.0e-10,
                             "absolute_tolerance": 1.0e-10,
                             "monitor_convergence": False}

        # employing Chebyshev for the (1,1)-block
        e_min_v = 0.3924
        e_max_v = 2.0598
        sp_11block = {
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            "ksp_chebyshev_eigenvalues": f"{e_min_v:.16e}, {e_max_v:.16e}",
            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
            "ksp_chebyshev_esteig_steps": 0,
            "ksp_chebyshev_esteig_noisy": False,
            "ksp_max_it": 20,
            "ksp_atol": 0.0,
            "ksp_rtol": 0.0}

        # employing Chebyshev for the pressure-mass matrix
        e_min_p = 0.5
        e_max_p = 2.0
        sp_M_p = {
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",
            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
            "ksp_chebyshev_esteig_steps": 0,
            "ksp_chebyshev_esteig_noisy": False,
            "ksp_max_it": 20,
            "ksp_atol": 0.0,
            "ksp_rtol": 0.0}

        auxiliary_sp = {"sp_11block": sp_11block,
                        "sp_M_p": sp_M_p}

        my_control_stationary.incompressible_linear_solve(
            ConstantNullspace(), solver_parameters=solver_parameters,
            auxiliary_sp=auxiliary_sp,
            print_error=False, outputs=False, plots=False)

        my_v = Function(space_0, name="my_v")
        my_zeta = Function(space_0, name="my_zeta")

        my_p = Function(space_1, name="my_p")
        my_mu = Function(space_1, name="my_mu")

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)

        my_p.assign(my_control_stationary._p)
        my_mu.assign(my_control_stationary._mu)

        v_ref = Function(
            VectorFunctionSpace(mesh, "Lagrange", degree + 2),
            name="v_ref")
        v_ref.interpolate(ref_sol_v(*X))

        v_error_norm = np.sqrt(abs(assemble(inner(my_v - v_ref,
                                                  my_v - v_ref) * dx)))
        print(f"{degree=} {p=} {N=} {v_error_norm=}")
        v_error_norms.append(v_error_norm)

        p_ref = Function(FunctionSpace(mesh, "Lagrange", degree + 1),
                         name="p_ref")
        p_ref.interpolate(ref_sol_p(*X))

        p_error_norm = np.sqrt(abs(assemble(inner(my_p - p_ref,
                                                  my_p - p_ref) * dx)))
        print(f"{degree=} {p=} {N=} {p_error_norm=}")
        p_error_norms.append(p_error_norm)

        zeta_ref = Function(
            VectorFunctionSpace(mesh, "Lagrange", degree + 2),
            name="zeta_ref")
        zeta_ref.interpolate(ref_sol_zeta(*X))

        zeta_error_norm = np.sqrt(abs(assemble(
            inner(my_zeta - zeta_ref,
                  my_zeta - zeta_ref) * dx)))
        print(f"{degree=} {p=} {N=} {zeta_error_norm=}")
        zeta_error_norms.append(zeta_error_norm)

        mu_ref = Function(FunctionSpace(mesh, "Lagrange", degree + 1),
                          name="mu_ref")
        mu_ref.interpolate(ref_sol_mu(*X))

        mu_error_norm = np.sqrt(abs(assemble(inner(my_mu - mu_ref,
                                                   my_mu - mu_ref) * dx)))
        print(f"{degree=} {p=} {N=} {mu_error_norm=}")
        mu_error_norms.append(mu_error_norm)

    v_error_norms = np.array(v_error_norms)
    v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {v_orders=}")
    min_v_ord = min(v_orders)
    assert min_v_ord > 1.5

    zeta_error_norms = np.array(zeta_error_norms)
    zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {zeta_orders=}")
    min_z_ord = min(zeta_orders)
    assert min_z_ord > 1.5

    p_error_norms = np.array(p_error_norms)
    p_orders = np.log(p_error_norms[:-1] / p_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {p_orders=}")
    min_p_ord = min(p_orders)
    assert min_p_ord > 1.5

    mu_error_norms = np.array(mu_error_norms)
    mu_orders = np.log(mu_error_norms[:-1] / mu_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {mu_orders=}")
    min_mu_ord = min(mu_orders)
    assert min_mu_ord > 1.5


@pytest.mark.parametrize("degree", tuple(range(1, 4)))
@pytest.mark.parametrize("nonlinear, Gauss_Newton", [(False, False), (True, False), (True, True)])
def test_stationary_control_with_reference_sol(degree, nonlinear, Gauss_Newton):
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    X = SpatialCoordinate(mesh)

    def alpha(v_old):
        if nonlinear:
            return Constant(2.0) + 0.5 * v_old**2.0
        else:
            return Constant(2.0)

    def forw_diff_operator(trial, test, v_old):
        # spatial differential for the forward problem
        return (
            inner(grad(trial), grad(test)) * dx
            + alpha(v_old) * inner(trial, test) * dx)

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1]))

        # true v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(0.0)

        return inner(f, test) * dx

    def reference(X):
        return sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1])

    space_0 = FunctionSpace(mesh, "Lagrange", degree)
    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    bc = DirichletBC(space_0, 0.0, "on_boundary")

    my_beta = 1.0

    my_control_stationary = Stationary(
        space_0, forw_diff_operator, desired_state=desired_state,
        force_function=force_f, beta=my_beta, bcs_v=bc,
        Gauss_Newton=Gauss_Newton)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-14,
                         "absolute_tolerance": 1.0e-14,
                         "monitor_convergence": False}

    if nonlinear:
        nl_sp = {"nl_max_it": 100,
                 "nl_rtol": 1.0e-10}

        my_control_stationary.non_linear_solve(
            solver_parameters=solver_parameters,
            nl_sp=nl_sp, outputs=False, plots=False)
    else:
        my_control_stationary.linear_solve(
            solver_parameters=solver_parameters,
            outputs=False,
            plots=False)

    my_v = Function(space_0)
    my_zeta = Function(space_0)
    my_control = Function(space_0)

    my_v.assign(my_control_stationary._v)
    my_zeta.assign(my_control_stationary._zeta)
    my_control.assign((1.0 / my_beta) * my_zeta)

    beta = 1.0

    def forward(u_ref, m):
        m_1 = Function(space_0, name="m_1")
        DirichletBC(space_0, m, "on_boundary").apply(m_1)
        m_0 = Function(space_0, name="m_0")
        m_0.assign(m - m_1)

        u = Function(space_0, name="u")
        solve(alpha(u) * inner(u, test_0) * dx
              + inner(grad(u), grad(test_0)) * dx
              - inner(m_0, test_0) * dx == 0,
              u, bc,
              solver_parameters={"snes_type": "newtonls",
                                 "snes_rtol": 1.0e-12,
                                 "snes_atol": 1.0e-15,
                                 "snes_stol": 0.0,
                                 "ksp_type": "preonly",
                                 "pc_type": "cholesky"})

        return assemble(inner(u - u_ref, u - u_ref) * dx
                        + beta * beta * inner(m_0, m_0) * dx
                        + inner(m_1, m_1) * ds)

    u_ref = Function(space_0, name="u_ref")
    u_ref.interpolate(reference(X))
    m0 = Function(space_0, name="m0")

    forward_J = partial(forward, u_ref)

    continue_annotation()
    J = forward_J(m0)
    pause_annotation()

    m = minimize(
        ReducedFunctional(J, Control_ad(m0, riesz_map="l2")),
        method="L-BFGS-B",
        options={"ftol": 0.0,
                 "gtol": 1.0e-9})

    dJ = compute_derivative(
        J, Control_ad(m0), apply_riesz=False)
    get_working_tape().clear_tape()

    dJ_dual = Function(space_0, name="dJ_dual")
    M_solver = LinearSolver(assemble(inner(trial_0, test_0) * dx),
                            solver_parameters={"ksp_type": "preonly",
                                               "pc_type": "cholesky"})
    M_solver.solve(dJ_dual, dJ.copy(deepcopy=True))
    dJ_norm = np.sqrt(abs(assemble(inner(dJ_dual, dJ_dual) * dx)))
    print(f"Gradient M^{-1} norm = {dJ_norm}")

    v_sol = Function(space_0, name="v_sol")
    solve(alpha(v_sol) * inner(v_sol, test_0) * dx
          + inner(grad(v_sol), grad(test_0)) * dx
          - inner(m, test_0) * dx == 0,
          v_sol, bc,
          solver_parameters={"snes_type": "newtonls",
                             "snes_rtol": 1.0e-12,
                             "snes_atol": 1.0e-15,
                             "snes_stol": 0.0,
                             "ksp_type": "preonly",
                             "pc_type": "cholesky"})

    v_error_norm = np.sqrt(abs(assemble(inner(my_v - v_sol,
                                              my_v - v_sol) * dx)))
    print(f"Error on the state: {v_error_norm}")
    assert v_error_norm < 1.0e-8

    control_error_norm = np.sqrt(abs(assemble(inner(my_control - m,
                                                    my_control - m) * dx)))
    print(f"Error on the control: {control_error_norm}")
    assert control_error_norm < 1.0e-6


def test_stationary_incompressible_non_linear_control():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

    space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
    my_DirichletBC_v = [DirichletBC(space_v, Constant((1.0, 0.0)), (4,)),
                        DirichletBC(space_v, 0.0, (1, 2, 3))]

    space_p = FunctionSpace(mesh, "Lagrange", 1)

    # defining the forward form
    def forw_diff_operator(trial, test, u):
        # viscosity
        nu = 1.0 / 100.0
        # spatial differential for the forward problem
        return (
            nu * inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), u), test) * dx)

    def desired_state(test):
        space = test.function_space()

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(as_vector([0.0, 0.0]))

        # true v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)
        f.interpolate(as_vector([0.0, 0.0]))

        return inner(f, test) * dx

    beta = 1.0e-3

    my_control_stationary = Stationary(
        space_v, forw_diff_operator, desired_state=desired_state,
        force_function=force_f, beta=beta, space_p=space_p,
        bcs_v=my_DirichletBC_v)

    # employing Chebyshev for the (1,1)-block
    e_min_v = 0.3924
    e_max_v = 2.0598
    sp_11block = {
        "ksp_type": "chebyshev",
        "pc_type": "jacobi",
        "ksp_chebyshev_eigenvalues": f"{e_min_v:.16e}, {e_max_v:.16e}",
        "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
        "ksp_chebyshev_esteig_steps": 0,
        "ksp_chebyshev_esteig_noisy": False,
        "ksp_max_it": 20,
        "ksp_atol": 0.0,
        "ksp_rtol": 0.0}

    # employing Chebyshev for the pressure-mass matrix
    e_min_p = 0.5
    e_max_p = 2.0
    sp_M_p = {
        "ksp_type": "chebyshev",
        "pc_type": "jacobi",
        "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",
        "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
        "ksp_chebyshev_esteig_steps": 0,
        "ksp_chebyshev_esteig_noisy": False,
        "ksp_max_it": 20,
        "ksp_atol": 0.0,
        "ksp_rtol": 0.0}

    auxiliary_sp = {"sp_11block": sp_11block,
                    "sp_M_p": sp_M_p}

    nl_sp = {"nl_max_it": 5,
             "nl_atol": 1.0e-8,
             "nl_rtol": 1.0e-5}

    my_control_stationary.incompressible_non_linear_solve(
        ConstantNullspace(), auxiliary_sp=auxiliary_sp,
        nl_sp=nl_sp, print_error=False,
        outputs=False, plots=False)

    my_v = Function(space_v)
    my_zeta = Function(space_v)

    my_v.assign(my_control_stationary._v)
    my_zeta.assign(my_control_stationary._zeta)


@pytest.mark.parametrize("degree", tuple(range(2, 3)))
def test_MMS_stationary_Navier_Stokes_control(degree):
    p_range = (2, 4)
    beta = 1.0e-3
    nu = 1.0 / 100.0

    def ref_sol_v(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        v = as_vector([X_1 * (X_2 ** 3), (1. / 4.) * (X_1 ** 4 - X_2 ** 4)])

        return v

    def ref_sol_zeta(x, y):
        zeta = as_vector([0.0, 0.0])

        return zeta

    def forw_diff_operator(trial, test, u):
        # spatial differential for the forward problem
        return (
            nu * inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), u), test) * dx)

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(v)

        # true v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def force_f(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X)

        # force function
        f = Function(space)
        f.interpolate(
            - 0.5 * nu * div(grad(v) + ufl.transpose(grad(v)))
            + grad(v) * v)

        return inner(f, test) * dx

    v_error_norms = []
    zeta_error_norms = []

    for p in range(*p_range):
        N = 2 ** p
        mesh = RectangleMesh(N, N, 2.0, 2.0)
        X = SpatialCoordinate(mesh)
        X_1 = X[0] - 1.0
        X_2 = X[1] - 1.0

        space_0 = VectorFunctionSpace(mesh, "Lagrange", degree)
        space_1 = FunctionSpace(mesh, "Lagrange", degree - 1)
        bc = DirichletBC(
            space_0,
            as_vector([
                X_1 * (X_2 ** 3),
                (1. / 4.) * (X_1 ** 4 - X_2 ** 4)]),
            "on_boundary")

        my_control_stationary = Stationary(
            space_0, forw_diff_operator, desired_state=desired_state,
            force_function=force_f, beta=beta, space_p=space_1, bcs_v=bc)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 500,
                             "relative_tolerance": 1.0e-10,
                             "absolute_tolerance": 1.0e-10,
                             "monitor_convergence": False}

        # employing Chebyshev for the (1,1)-block
        e_min_v = 0.3924
        e_max_v = 2.0598
        sp_11block = {
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            "ksp_chebyshev_eigenvalues": f"{e_min_v:.16e}, {e_max_v:.16e}",
            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
            "ksp_chebyshev_esteig_steps": 0,
            "ksp_chebyshev_esteig_noisy": False,
            "ksp_max_it": 20,
            "ksp_atol": 0.0,
            "ksp_rtol": 0.0}

        # employing Chebyshev for the pressure-mass matrix
        e_min_p = 0.5
        e_max_p = 2.0
        sp_M_p = {
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",
            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
            "ksp_chebyshev_esteig_steps": 0,
            "ksp_chebyshev_esteig_noisy": False,
            "ksp_max_it": 20,
            "ksp_atol": 0.0,
            "ksp_rtol": 0.0}

        auxiliary_sp = {"sp_11block": sp_11block,
                        "sp_M_p": sp_M_p}

        nl_sp = {"nl_max_it": 10,
                 "nl_atol": 1.0e-9,
                 "nl_rtol": 1.0e-9}

        my_control_stationary.incompressible_non_linear_solve(
            ConstantNullspace(), solver_parameters=solver_parameters,
            auxiliary_sp=auxiliary_sp,
            nl_sp=nl_sp, print_error=False,
            outputs=False, plots=False)

        my_v = Function(space_0, name="my_v")
        my_zeta = Function(space_0, name="my_zeta")

        my_p = Function(space_1, name="my_p")
        my_mu = Function(space_1, name="my_mu")

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)

        my_p.assign(my_control_stationary._p)
        my_mu.assign(my_control_stationary._mu)

        v_ref = Function(VectorFunctionSpace(mesh, "Lagrange", degree + 2),
                         name="v_ref")
        v_ref.interpolate(ref_sol_v(*X))

        v_error_norm = np.sqrt(abs(assemble(inner(my_v - v_ref,
                                                  my_v - v_ref) * dx)))
        print(f"{degree=} {p=} {N=} {v_error_norm=}")
        v_error_norms.append(v_error_norm)

        zeta_ref = Function(
            VectorFunctionSpace(mesh, "Lagrange", degree + 2),
            name="zeta_ref")
        zeta_ref.interpolate(ref_sol_zeta(*X))

        zeta_error_norm = np.sqrt(abs(assemble(
            inner(my_zeta - zeta_ref,
                  my_zeta - zeta_ref) * dx)))
        print(f"{degree=} {p=} {N=} {zeta_error_norm=}")
        zeta_error_norms.append(zeta_error_norm)

    v_error_norms = np.array(v_error_norms)
    v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {v_orders=}")
    min_v_ord = min(v_orders)
    assert min_v_ord > 1.5

    zeta_error_norms = np.array(zeta_error_norms)
    zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)
    print(f"{degree=} {zeta_orders=}")
    min_z_ord = min(zeta_orders)
    assert min_z_ord > 1.5

