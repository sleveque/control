#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *

from tlm_adjoint.firedrake import *

from preconditioner import *

from control import *

import mpi4py.MPI as MPI
import numpy as np
import ufl
import pytest

stop_manager()


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


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
        v_d.interpolate(cos(pi * X[0] - 0.5 * pi) * cos(pi * X[1] - 0.5 * pi) + 1.)  # noqa: E501

        return inner(v_d, test) * dx, v_d

    def force_f(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # force function
        f = Function(space)

        f.interpolate(2.0 * pi**2 * cos(pi * X[0] - 0.5 * pi) * cos(pi * X[1] - 0.5 * pi))  # noqa: E501

        return inner(f, test) * dx

    beta = 10.0**-3

    my_control_stationary = Control.Stationary(
        space_0, forw_diff_operator, desired_state, force_f, beta=beta,
        bcs_v=None)

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

    # employing Chebyshev for the (1,1)-block
    lambda_v_bounds = (0.25, 1.5625)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-14,
                         "absolute_tolerance": 1.0e-14,
                         "monitor_convergence": False}

    my_control_stationary.linear_solve(
        lambda_v_bounds=lambda_v_bounds,
        solver_parameters=solver_parameters,
        v_d=b_0, f=b_1, create_output=False,
        plots=False)

    v_0.assign(my_control_stationary._v)
    zeta_0.assign(my_control_stationary._zeta)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-13
    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-13


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
        v_d.interpolate((0.0, 0.0))

        return inner(v_d, test) * dx, v_d

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate((0.0, 0.0))

        return inner(f, test) * dx

    beta = 10.0**-3

    my_control_stationary = Control.Stationary(
        space_0, forw_diff_operator, desired_state, force_f, beta=beta,
        space_p=space_1, bcs_v=DirichletBC(space_0, 0.0, "on_boundary"))

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

    # employing Chebyshev for the (1,1)-block
    lambda_v_bounds = (0.25, 1.5625)
    lambda_p_bounds = (0.25, 2.25)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-15,
                         "absolute_tolerance": 1.0e-15,
                         "monitor_convergence": False}

    my_control_stationary.incompressible_linear_solve(
        ConstantNullspace(), solver_parameters=solver_parameters,
        lambda_v_bounds=lambda_v_bounds, lambda_p_bounds=lambda_p_bounds,
        v_d=b_0, f=b_1, div_v=b_2, div_zeta=b_3,
        print_error=False, create_output=False,
        plots=False)

    v_0.assign(my_control_stationary._v)
    zeta_0.assign(my_control_stationary._zeta)
    p_0.assign(my_control_stationary._p)
    mu_0.assign(my_control_stationary._mu)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-13
    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-13

    p_error_norm = np.sqrt(abs(assemble(inner(p_0 - p_ref,
                                              p_0 - p_ref) * dx)))
    assert p_error_norm < 1.0e-13
    mu_error_norm = np.sqrt(abs(assemble(inner(mu_0 - mu_ref,
                                               mu_0 - mu_ref) * dx)))
    assert mu_error_norm < 1.0e-13


def test_stationary_linear_control_with_reference_sol():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    X = SpatialCoordinate(mesh)

    def forw_diff_operator(trial, test, v_old):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx + 2.0 * inner(trial, test) * dx  # noqa: E501

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1]))

        return inner(v_d, test) * dx, v_d

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    def alpha_linear(v_old):
        return Constant(2.0)

    def reference(X):
        return sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1])

    for degree in range(1, 4):
        space_0 = FunctionSpace(mesh, "Lagrange", degree)
        test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
        bc = DirichletBC(space_0, 0.0, "on_boundary")

        my_beta = 1.0

        my_control_stationary = Control.Stationary(
            space_0, forw_diff_operator, desired_state, force_f, beta=my_beta,
            bcs_v=bc)

        # employing Chebyshev for the (1,1)-block
        lambda_v_bounds = (0.25, 2.25)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 500,
                             "relative_tolerance": 1.0e-14,
                             "absolute_tolerance": 1.0e-14,
                             "monitor_convergence": False}

        my_control_stationary.linear_solve(
            lambda_v_bounds=lambda_v_bounds,
            solver_parameters=solver_parameters,
            create_output=False,
            plots=False)

        my_v = Function(space_0)
        my_zeta = Function(space_0)
        my_control = Function(space_0)

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)
        my_control.assign((1.0 / my_beta) * my_zeta)

        del my_control_stationary

        beta = Constant(1.0)

        def forward(u_ref, m):
            m_1 = Function(space_0, name="m_1")
            DirichletBCSolver(m, m_1, "on_boundary").solve()
            m_0 = Function(space_0, name="m_0")
            m_0.assign(m - m_1)

            u = Function(space_0, name="u")
            solve(alpha_linear(u) * inner(u, test_0) * dx
                  + inner(grad(u), grad(test_0)) * dx
                  - inner(m_0, test_0) * dx == 0,
                  u, bc,
                  solver_parameters={"snes_type": "newtonls",
                                     "snes_rtol": 1.0e-12,
                                     "snes_atol": 1.0e-15,
                                     "snes_stol": 0.0,
                                     "ksp_type": "preonly",
                                     "pc_type": "cholesky"})

            J = Functional(name="J")
            J.assign(inner(u - u_ref, u - u_ref) * dx
                     + beta * beta * inner(m_0, m_0) * dx
                     + inner(m_1, m_1) * ds)
            return J

        u_ref = Function(space_0, name="u_ref")
        u_ref.interpolate(reference(X))
        m0 = Function(space_0, name="m0")

        def forward_J(m):
            return forward(u_ref, m)

        m, result = minimize_scipy(forward_J, m0,
                                   method="L-BFGS-B",
                                   options={"ftol": 0.0,
                                            "gtol": 1.0e-8,
                                            "disp": False})
        assert result.success

        m_ = m.copy(deepcopy=True)
        start_manager()
        J = forward_J(m_)
        stop_manager()
        dJ = compute_gradient(J, m_)
        reset_manager()
        stop_manager()
        del m_

        dJ_dual = Function(space_0, name="dJ_dual")
        M_solver = LinearSolver(assemble(inner(trial_0, test_0) * dx),
                                solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "cholesky"})
        M_solver.solve(dJ_dual, dJ.copy(deepcopy=True))
        dJ_norm = np.sqrt(abs(assemble(inner(dJ_dual, dJ_dual) * dx)))
        print(f"Gradient M^{-1} norm = {dJ_norm}")

        v_sol = Function(space_0, name="v_sol")
        solve(alpha_linear(v_sol) * inner(v_sol, test_0) * dx
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


def test_Picard_stationary_non_linear_control_with_reference_sol():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    X = SpatialCoordinate(mesh)

    def forw_diff_operator(trial, test, v_old):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx + (Constant(2.0) + 0.5 * v_old**2.0) * inner(trial, test) * dx  # noqa: E501

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1]))

        return inner(v_d, test) * dx, v_d

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    def alpha_non_linear(v_old):
        return Constant(2.0) + 0.5 * v_old**2.0

    def reference(X):
        return sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1])

    for degree in range(1, 4):
        space_0 = FunctionSpace(mesh, "Lagrange", degree)
        test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
        bc = DirichletBC(space_0, 0.0, "on_boundary")

        my_beta = 1.0

        my_control_stationary = Control.Stationary(
            space_0, forw_diff_operator, desired_state, force_f, beta=my_beta,
            bcs_v=bc)

        # employing Chebyshev for the (1,1)-block
        lambda_v_bounds = (0.25, 2.25)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 500,
                             "relative_tolerance": 1.0e-14,
                             "absolute_tolerance": 1.0e-14,
                             "monitor_convergence": False}

        my_control_stationary.non_linear_solve(
            lambda_v_bounds=lambda_v_bounds,
            solver_parameters=solver_parameters,
            max_non_linear_iter=100, relative_non_linear_tol=10.0**-10,
            print_error_linear=False, create_output=False,
            plots=False)

        my_v = Function(space_0)
        my_zeta = Function(space_0)
        my_control = Function(space_0)

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)
        my_control.assign((1.0 / my_beta) * my_zeta)

        del my_control_stationary

        beta = Constant(1.0)

        def forward(u_ref, m):
            m_1 = Function(space_0, name="m_1")
            DirichletBCSolver(m, m_1, "on_boundary").solve()
            m_0 = Function(space_0, name="m_0")
            m_0.assign(m - m_1)

            u = Function(space_0, name="u")
            solve(alpha_non_linear(u) * inner(u, test_0) * dx
                  + inner(grad(u), grad(test_0)) * dx
                  - inner(m_0, test_0) * dx == 0,
                  u, bc,
                  solver_parameters={"snes_type": "newtonls",
                                     "snes_rtol": 1.0e-12,
                                     "snes_atol": 1.0e-15,
                                     "snes_stol": 0.0,
                                     "ksp_type": "preonly",
                                     "pc_type": "cholesky"})

            J = Functional(name="J")
            J.assign(inner(u - u_ref, u - u_ref) * dx
                     + beta * beta * inner(m_0, m_0) * dx
                     + inner(m_1, m_1) * ds)
            return J

        u_ref = Function(space_0, name="u_ref")
        u_ref.interpolate(reference(X))
        m0 = Function(space_0, name="m0")

        def forward_J(m):
            return forward(u_ref, m)

        m, result = minimize_scipy(forward_J, m0,
                                   method="L-BFGS-B",
                                   options={"ftol": 0.0,
                                            "gtol": 1.0e-8,
                                            "disp": False})
        assert result.success

        m_ = m.copy(deepcopy=True)
        start_manager()
        J = forward_J(m_)
        stop_manager()
        dJ = compute_gradient(J, m_)
        reset_manager()
        stop_manager()
        del m_

        dJ_dual = Function(space_0, name="dJ_dual")
        M_solver = LinearSolver(assemble(inner(trial_0, test_0) * dx),
                                solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "cholesky"})
        M_solver.solve(dJ_dual, dJ.copy(deepcopy=True))
        dJ_norm = np.sqrt(abs(assemble(inner(dJ_dual, dJ_dual) * dx)))
        print(f"Gradient M^{-1} norm = {dJ_norm}")

        v_sol = Function(space_0, name="v_sol")
        solve(alpha_non_linear(v_sol) * inner(v_sol, test_0) * dx
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


def test_GN_stationary_non_linear_control_with_reference_sol():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    X = SpatialCoordinate(mesh)

    def forw_diff_operator(trial, test, v_old):
        # spatial differential for the forward problem
        return inner(grad(v_old), grad(test)) * dx + (Constant(2.0) + 0.5 * v_old**2.0) * inner(v_old, test) * dx  # noqa: E501

    def desired_state(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1]))

        return inner(v_d, test) * dx, v_d

    def force_f(test):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    def alpha_non_linear(v_old):
        return Constant(2.0) + 0.5 * v_old**2.0

    def reference(X):
        return sin(pi * X[0]) * sin(pi * X[1]) * exp(X[0] + X[1])

    for degree in range(1, 4):
        space_0 = FunctionSpace(mesh, "Lagrange", degree)
        test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
        bc = DirichletBC(space_0, 0.0, "on_boundary")

        my_beta = 1.0

        my_control_stationary = Control.Stationary(
            space_0, forw_diff_operator, desired_state, force_f, beta=my_beta,
            bcs_v=bc)

        # employing Gauss-Newton
        my_control_stationary.set_Gauss_Newton()

        # employing Chebyshev for the (1,1)-block
        lambda_v_bounds = (0.25, 2.25)

        solver_parameters = {"linear_solver": "fgmres",
                             "fgmres_restart": 10,
                             "maximum_iterations": 500,
                             "relative_tolerance": 1.0e-14,
                             "absolute_tolerance": 1.0e-14,
                             "monitor_convergence": False}

        my_control_stationary.non_linear_solve(
            lambda_v_bounds=lambda_v_bounds,
            solver_parameters=solver_parameters,
            max_non_linear_iter=100, relative_non_linear_tol=10.0**-8,
            print_error_linear=False, create_output=False,
            plots=False)

        my_v = Function(space_0)
        my_zeta = Function(space_0)
        my_control = Function(space_0)

        my_v.assign(my_control_stationary._v)
        my_zeta.assign(my_control_stationary._zeta)
        my_control.assign((1.0 / my_beta) * my_zeta)

        del my_control_stationary

        beta = Constant(1.0)

        def forward(u_ref, m):
            m_1 = Function(space_0, name="m_1")
            DirichletBCSolver(m, m_1, "on_boundary").solve()
            m_0 = Function(space_0, name="m_0")
            m_0.assign(m - m_1)

            u = Function(space_0, name="u")
            solve(alpha_non_linear(u) * inner(u, test_0) * dx
                  + inner(grad(u), grad(test_0)) * dx
                  - inner(m_0, test_0) * dx == 0,
                  u, bc,
                  solver_parameters={"snes_type": "newtonls",
                                     "snes_rtol": 1.0e-12,
                                     "snes_atol": 1.0e-15,
                                     "snes_stol": 0.0,
                                     "ksp_type": "preonly",
                                     "pc_type": "cholesky"})

            J = Functional(name="J")
            J.assign(inner(u - u_ref, u - u_ref) * dx
                     + beta * beta * inner(m_0, m_0) * dx
                     + inner(m_1, m_1) * ds)
            return J

        u_ref = Function(space_0, name="u_ref")
        u_ref.interpolate(reference(X))
        m0 = Function(space_0, name="m0")

        def forward_J(m):
            return forward(u_ref, m)

        m, result = minimize_scipy(forward_J, m0,
                                   method="L-BFGS-B",
                                   options={"ftol": 0.0,
                                            "gtol": 1.0e-8,
                                            "disp": False})
        assert result.success

        m_ = m.copy(deepcopy=True)
        start_manager()
        J = forward_J(m_)
        stop_manager()
        dJ = compute_gradient(J, m_)
        reset_manager()
        stop_manager()
        del m_

        dJ_dual = Function(space_0, name="dJ_dual")
        M_solver = LinearSolver(assemble(inner(trial_0, test_0) * dx),
                                solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "cholesky"})
        M_solver.solve(dJ_dual, dJ.copy(deepcopy=True))
        dJ_norm = np.sqrt(abs(assemble(inner(dJ_dual, dJ_dual) * dx)))
        print(f"Gradient M^{-1} norm = {dJ_norm}")

        v_sol = Function(space_0, name="v_sol")
        solve(alpha_non_linear(v_sol) * inner(v_sol, test_0) * dx
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
        nu = 1.0 / 500.0
        # spatial differential for the forward problem
        return nu * inner(grad(trial), grad(test)) * dx + inner(dot(u, grad(trial)), test) * dx  # noqa: E501

    def desired_state_vec(test):
        space = test.function_space()

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(Constant((0.0, 0.0)))

        return inner(v_d, test) * dx, v_d

    def force_f_vec(test):
        space = test.function_space()

        # force function
        f = Function(space)
        f.interpolate(Constant((0.0, 0.0)))

        return inner(f, test) * dx

    beta = 10.0**-3

    my_control_stationary = Control.Stationary(
        space_v, forw_diff_operator, desired_state_vec,
        force_f_vec, beta=beta, space_p=space_p,
        bcs_v=my_DirichletBC_v)

    lambda_v_bounds = (0.3924, 2.0598)
    lambda_p_bounds = (0.5, 2.0)

    my_control_stationary.incompressible_non_linear_solve(
        ConstantNullspace(), lambda_v_bounds=lambda_v_bounds,
        lambda_p_bounds=lambda_p_bounds, max_non_linear_iter=5,
        relative_non_linear_tol=10.0**-5, absolute_non_linear_tol=10.0**-8,
        print_error_linear=False, print_error_non_linear=False,
        create_output=False, plots=False)

    my_v = Function(space_v)
    my_zeta = Function(space_v)

    my_v.assign(my_control_stationary._v)
    my_zeta.assign(my_control_stationary._zeta)

    my_v_norm = np.sqrt(abs(assemble(inner(my_v, my_v) * dx)))
    my_zeta_norm = np.sqrt(abs(assemble(inner(my_zeta, my_zeta) * dx)))

    assert my_v_norm < 0.13
    assert my_zeta_norm < 0.0005


def test_instationary_linear_control_BE():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 2)

    test_0 = TestFunction(space_0)

    def forw_diff_operator(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test, t):
        space = test.function_space()

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(Constant(0.0))

        return inner(v_d, test) * dx, v_d

    def force_f(test, t):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    beta = 10**-3

    n_t = 5
    tau = 0.25

    def my_DirichletBC_t(space_0, t):
        my_DirichletBC = DirichletBC(space_0, 0.0, "on_boundary")
        return my_DirichletBC

    my_control_instationary = Control.Instationary(
        space_0, forw_diff_operator, desired_state, force_f,
        beta=beta, time_interval=(0.0, 1.0),
        CN=False, n_t=n_t, bcs_v=my_DirichletBC_t)

    flattened_space_v = tuple(space_0 for i in range(n_t))
    mixed_element_v = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_v])
    full_space_v = FunctionSpace(space_0.mesh(), mixed_element_v)

    v_0 = Function(full_space_v)
    zeta_0 = Function(full_space_v)
    v_ref = Function(full_space_v)
    zeta_ref = Function(full_space_v)

    v_ref.sub(0).interpolate(Constant(0.0))
    v_ref.sub(1).interpolate(
        tau * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    v_ref.sub(2).interpolate(
        (tau**2.0) * X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    v_ref.sub(3).interpolate(
        (tau**3.0) * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    v_ref.sub(4).interpolate(
        (tau**4.0) * X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    zeta_ref.sub(0).interpolate(
        sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    zeta_ref.sub(1).interpolate(
        tau * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    zeta_ref.sub(2).interpolate(
        (tau**2.0) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    zeta_ref.sub(3).interpolate(
        (tau**3.0) * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    zeta_ref.sub(4).interpolate(Constant(0.0))

    v_0 = Function(full_space_v)
    zeta_0 = Function(full_space_v)

    b_0 = Function(full_space_v)
    b_1 = Function(full_space_v)

    b_v_help_0 = Function(space_0)
    b_v_help_1 = Function(space_0)
    b_zeta_help_0 = Function(space_0)
    b_zeta_help_1 = Function(space_0)

    b_v_help_0.assign(v_ref.sub(0))
    b_zeta_help_0.assign(zeta_ref.sub(0))
    b_zeta_help_1.assign(zeta_ref.sub(1))
    b_0_help = assemble(
        tau * inner(b_v_help_0, test_0) * dx
        + tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(1))
    b_zeta_help_1.assign(zeta_ref.sub(2))
    b_1_help = assemble(
        tau * inner(b_v_help_0, test_0) * dx
        + tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(2))
    b_zeta_help_0.assign(zeta_ref.sub(2))
    b_zeta_help_1.assign(zeta_ref.sub(3))
    b_2_help = assemble(
        tau * inner(b_v_help_0, test_0) * dx
        + tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(3))
    b_zeta_help_0.assign(zeta_ref.sub(3))
    b_zeta_help_1.assign(zeta_ref.sub(4))
    b_3_help = assemble(
        tau * inner(b_v_help_0, test_0) * dx
        + tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_zeta_help_0.assign(zeta_ref.sub(4))
    b_4_help = assemble(
        tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx)
    b_0.sub(0).assign(assemble(b_0_help))
    b_0.sub(1).assign(assemble(b_1_help))
    b_0.sub(2).assign(assemble(b_2_help))
    b_0.sub(3).assign(assemble(b_3_help))
    b_0.sub(4).assign(assemble(b_4_help))

    b_v_help_0.assign(v_ref.sub(0))
    b_0_help = assemble(
        tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(1))
    b_v_help_1.assign(v_ref.sub(0))
    b_zeta_help_0.assign(zeta_ref.sub(1))
    b_1_help = assemble(
        tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        - inner(b_v_help_1, test_0) * dx
        - (tau / beta) * inner(b_zeta_help_0, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(2))
    b_v_help_1.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(2))
    b_2_help = assemble(
        tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        - inner(b_v_help_1, test_0) * dx
        - (tau / beta) * inner(b_zeta_help_0, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(3))
    b_v_help_1.assign(v_ref.sub(2))
    b_zeta_help_0.assign(zeta_ref.sub(3))
    b_3_help = assemble(
        tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        - inner(b_v_help_1, test_0) * dx
        - (tau / beta) * inner(b_zeta_help_0, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(4))
    b_v_help_1.assign(v_ref.sub(3))
    b_zeta_help_0.assign(zeta_ref.sub(4))
    b_4_help = assemble(
        tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        - inner(b_v_help_1, test_0) * dx
        - (tau / beta) * inner(b_zeta_help_0, test_0) * dx)
    b_1.sub(0).assign(assemble(b_0_help))
    b_1.sub(1).assign(assemble(b_1_help))
    b_1.sub(2).assign(assemble(b_2_help))
    b_1.sub(3).assign(assemble(b_3_help))
    b_1.sub(4).assign(assemble(b_4_help))

    # employing Chebyshev for the (1,1)-block
    lambda_v_bounds = (0.25, 1.5625)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-14,
                         "absolute_tolerance": 1.0e-14,
                         "monitor_convergence": False}

    my_control_instationary.linear_solve(
        lambda_v_bounds=lambda_v_bounds,
        solver_parameters=solver_parameters,
        v_d=b_0, f=b_1,
        print_error=False, create_output=False,
        plots=False)

    v_0.assign(my_control_instationary._v)
    zeta_0.assign(my_control_instationary._zeta)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-13

    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-13


def test_instationary_linear_control_CN():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 2)

    test_0 = TestFunction(space_0)

    def forw_diff_operator(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test, t):
        space = test.function_space()

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(Constant(0.0))

        return inner(v_d, test) * dx, v_d

    def force_f(test, t):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    beta = 10**-3

    n_t = 5
    tau = 0.25

    def my_DirichletBC_t(space_0, t):
        my_DirichletBC = DirichletBC(space_0, 0.0, "on_boundary")
        return my_DirichletBC

    my_control_instationary = Control.Instationary(
        space_0, forw_diff_operator, desired_state, force_f,
        beta=beta, time_interval=(0.0, 1.0),
        CN=True, n_t=n_t, bcs_v=my_DirichletBC_t)

    flattened_space_v = tuple(space_0 for i in range(n_t))
    mixed_element_v = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_v])
    full_space_v = FunctionSpace(space_0.mesh(), mixed_element_v)

    flattened_space_v_CN = tuple(space_0 for i in range(n_t - 1))
    mixed_element_v_CN = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_v_CN])
    full_space_v_CN = FunctionSpace(space_0.mesh(), mixed_element_v_CN)

    v_0 = Function(full_space_v)
    zeta_0 = Function(full_space_v)
    v_ref = Function(full_space_v)
    zeta_ref = Function(full_space_v)

    v_ref.sub(0).interpolate(Constant(0.0))
    v_ref.sub(1).interpolate(
        tau * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    v_ref.sub(2).interpolate(
        (tau**2.0) * X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    v_ref.sub(3).interpolate(
        (tau**3.0) * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    v_ref.sub(4).interpolate(
        (tau**4.0) * X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    zeta_ref.sub(0).interpolate(
        sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    zeta_ref.sub(1).interpolate(
        tau * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    zeta_ref.sub(2).interpolate(
        (tau**2.0) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    zeta_ref.sub(3).interpolate(
        (tau**3.0) * sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))
    zeta_ref.sub(4).interpolate(Constant(0.0))

    v_0 = Function(full_space_v)
    zeta_0 = Function(full_space_v)

    b_0 = Function(full_space_v_CN)
    b_1 = Function(full_space_v_CN)

    b_v_help_0 = Function(space_0)
    b_v_help_1 = Function(space_0)
    b_zeta_help_0 = Function(space_0)
    b_zeta_help_1 = Function(space_0)

    b_v_help_0.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(0))
    b_zeta_help_1.assign(zeta_ref.sub(1))
    b_0_help = assemble(
        0.5 * tau * inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_1), grad(test_0)) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(2))
    b_v_help_1.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(1))
    b_zeta_help_1.assign(zeta_ref.sub(2))
    b_1_help = assemble(
        0.5 * tau * inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(b_v_help_1, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_1), grad(test_0)) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(3))
    b_v_help_1.assign(v_ref.sub(2))
    b_zeta_help_0.assign(zeta_ref.sub(2))
    b_zeta_help_1.assign(zeta_ref.sub(3))
    b_2_help = assemble(
        0.5 * tau * inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(b_v_help_1, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_1), grad(test_0)) * dx
        - inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(4))
    b_v_help_1.assign(v_ref.sub(3))
    b_zeta_help_0.assign(zeta_ref.sub(3))
    b_3_help = assemble(
        0.5 * tau * inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(b_v_help_1, test_0) * dx
        + 0.5 * tau * inner(grad(b_zeta_help_0), grad(test_0)) * dx
        + inner(b_zeta_help_0, test_0) * dx)
    b_0.sub(0).assign(assemble(b_0_help))
    b_0.sub(1).assign(assemble(b_1_help))
    b_0.sub(2).assign(assemble(b_2_help))
    b_0.sub(3).assign(assemble(b_3_help))

    b_v_help_0.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(0))
    b_zeta_help_1.assign(zeta_ref.sub(1))
    b_0_help = assemble(
        0.5 * tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_0, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(2))
    b_v_help_1.assign(v_ref.sub(1))
    b_zeta_help_0.assign(zeta_ref.sub(1))
    b_zeta_help_1.assign(zeta_ref.sub(2))
    b_1_help = assemble(
        0.5 * tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_v_help_1), grad(test_0)) * dx
        - inner(b_v_help_1, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_0, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(3))
    b_v_help_1.assign(v_ref.sub(2))
    b_zeta_help_0.assign(zeta_ref.sub(2))
    b_zeta_help_1.assign(zeta_ref.sub(3))
    b_2_help = assemble(
        0.5 * tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_v_help_1), grad(test_0)) * dx
        - inner(b_v_help_1, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_0, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_1, test_0) * dx)
    b_v_help_0.assign(v_ref.sub(4))
    b_v_help_1.assign(zeta_ref.sub(3))
    b_zeta_help_0.assign(zeta_ref.sub(3))
    b_3_help = assemble(
        0.5 * tau * inner(grad(b_v_help_0), grad(test_0)) * dx
        + inner(b_v_help_0, test_0) * dx
        + 0.5 * tau * inner(grad(b_v_help_1), grad(test_0)) * dx
        - inner(b_v_help_1, test_0) * dx
        - 0.5 * (tau / beta) * inner(b_zeta_help_0, test_0) * dx)
    b_1.sub(0).assign(assemble(b_0_help))
    b_1.sub(1).assign(assemble(b_1_help))
    b_1.sub(2).assign(assemble(b_2_help))
    b_1.sub(3).assign(assemble(b_3_help))

    # employing Chebyshev for the (1,1)-block
    lambda_v_bounds = (0.25, 1.5625)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 500,
                         "relative_tolerance": 1.0e-14,
                         "absolute_tolerance": 1.0e-14,
                         "monitor_convergence": False}

    my_control_instationary.linear_solve(
        lambda_v_bounds=lambda_v_bounds,
        solver_parameters=solver_parameters,
        v_d=b_0, f=b_1,
        print_error=False, create_output=False,
        plots=False)

    v_0.assign(my_control_instationary._v)
    zeta_0.assign(my_control_instationary._zeta)

    v_error_norm = np.sqrt(abs(assemble(inner(v_0 - v_ref,
                                              v_0 - v_ref) * dx)))
    assert v_error_norm < 1.0e-13

    zeta_error_norm = np.sqrt(abs(assemble(inner(zeta_0 - zeta_ref,
                                                 zeta_0 - zeta_ref) * dx)))
    assert zeta_error_norm < 1.0e-13


def test_instationary_Stokes_control_BE_with_exact_sol():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)

    space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_p = FunctionSpace(mesh, "Lagrange", 1)

    n_t = 10
    time_interval = (0.0, 1.0)

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)

        T_f = 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                exp(T_f - t) * 20.0 * X[0] * (X[1]**3),
                exp(T_f - t) * 5.0 * (X[0]**4 - X[1]**4)]),
            "on_boundary")

        return my_bcs

    beta = 10.0**-3

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.
        beta = 10.0**-3

        v_d_help = Function(space)
        v_d_help.interpolate(as_vector([
            4. * beta * X[1] * (2. * (3. * X[0] * X[0] - 1.) * (X[1] * X[1] - 1.) + 3. * (X[0] * X[0] - 1.)**2),  # noqa: E501
            -4. * beta * X[0] * (3. * (X[1] * X[1] - 1.)**2 + 2. * (X[0] * X[0] - 1.) * (3. * X[1] * X[1] - 1.))]))  # noqa: E501

        # desired state
        v_d = Function(space)
        v_d.interpolate(as_vector([
            exp(T_f - t) * (20. * X[0] * X[1]**3 + 2. * beta * (((X[0] * X[0] - 1.)**2) * (X[1] * X[1] - 7.) - 4. * (3. * X[0] * X[0] - 1.) * (X[1] * X[1] - 1.) + 2.)),  # noqa: E501
            exp(T_f - t) * (5. * (X[0]**4 - X[1]**4) - 2. * beta * X[0] * (((X[1] * X[1] - 1.)**2) * (X[0] * X[0] - 7.) - 4. * (X[0] * X[0] - 1.) * (3. * X[1] * X[1] - 1.) - 2.))]))  # noqa: E501

        with v_d.dat.vec as b_v, \
                v_d_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.
        t = 0.0

        v_0 = Function(space)
        v_0.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        # force function
        f_help = Function(space)
        f_help.interpolate(as_vector([
            2. * X[1] * (X[0]**2 - 1.)**2 * (X[1]**2 - 1.),
            -2. * X[0] * (X[0]**2 - 1.) * (X[1]**2 - 1.)**2]))

        # force function
        f = Function(space)
        f.interpolate(as_vector([
            exp(T_f - t) * (-20. * X[0] * X[1]**3 - 2. * X[1] * (X[0] * X[0] - 1.)**2 * (X[1] * X[1] - 1.)),  # noqa: E501
            exp(T_f - t) * (5. * (X[1]**4 - X[0]**4) + 2. * X[0] * (X[0] * X[0] - 1.) * (X[1] * X[1] - 1.)**2)]))  # noqa: E501

        with f.dat.vec as b_v, \
                f_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        return inner(f, test) * dx

    my_control_instationary = Control.Instationary(
        space_v, forw_diff_operator_v, desired_state_v, force_f_v,
        beta=beta, initial_condition=initial_condition_v,
        time_interval=time_interval, CN=False, n_t=n_t,
        bcs_v=my_DirichletBC_t_v)

    lambda_v_bounds = (0.25, 1.5625)
    lambda_p_bounds = (0.25, 2.25)

    my_control_instationary.incompressible_linear_solve(
        ConstantNullspace(), space_p=space_p, lambda_v_bounds=lambda_v_bounds,
        lambda_p_bounds=lambda_p_bounds, print_error=False,
        create_output=False)

    my_v = my_control_instationary._v
    my_p = my_control_instationary._p

    my_zeta = my_control_instationary._zeta
    my_mu = my_control_instationary._mu

    def v_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return true_v

    def p_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        true_p = Function(space)
        true_p.interpolate(
            exp(T_f - t) * (60. * X[0]**2 * X[1] - 20. * X[1]**3))

        return true_p

    def zeta_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        beta = 10.0**-3

        true_zeta = Function(space)
        true_zeta.interpolate(as_vector([
            beta * (exp(T_f - t) - 1.) * 2. * X[1] * (X[0]**2 - 1.)**2 * (X[1]**2 - 1.),  # noqa: E501
            -beta * (exp(T_f - t) - 1.) * 2. * X[0] * (X[0]**2 - 1.) * (X[1]**2 - 1.)**2]))  # noqa: E501

        return true_zeta

    def mu_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        beta = 10.0**-3

        true_mu = Function(space)
        true_mu.interpolate(
            beta * exp(T_f - t) * 4. * X[0] * X[1])

        return true_mu

    flattened_space_v = tuple(space_v for i in range(n_t))
    mixed_element_v = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_v])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

    flattened_space_p = tuple(space_p for i in range(n_t))
    mixed_element_p = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_p])
    full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

    true_v = Function(full_space_v, name="true_v")
    true_zeta = Function(full_space_v, name="true_zeta")

    true_p = Function(full_space_p, name="true_p")
    true_mu = Function(full_space_p, name="true_mu")

    t_0 = time_interval[0]
    T_f = time_interval[1]
    tau = (T_f - t_0) / (n_t - 1.0)

    v_test = TestFunction(space_v)
    p_test = TestFunction(space_p)

    for i in range(n_t):
        t = t_0 + i * tau

        true_v_i = v_sol(v_test, t)
        true_v.sub(i).assign(true_v_i)

        true_zeta_i = zeta_sol(v_test, t)
        true_zeta.sub(i).assign(true_zeta_i)

        true_p_i = p_sol(p_test, t)
        true_p.sub(i).assign(true_p_i)

        true_mu_i = mu_sol(p_test, t)
        true_mu.sub(i).assign(true_mu_i)

    v_error_norm = np.sqrt(abs(assemble(inner(my_v - true_v,
                                              my_v - true_v) * dx)))
    assert v_error_norm < 0.01

    zeta_error_norm = np.sqrt(abs(assemble(inner(my_zeta - true_zeta,
                                                 my_zeta - true_zeta) * dx)))
    assert zeta_error_norm < 0.01

    p_error_norm = np.sqrt(abs(assemble(inner(my_p - true_p,
                                              my_p - true_p) * dx)))
    assert p_error_norm < 0.01

    mu_error_norm = np.sqrt(abs(assemble(inner(my_mu - true_mu,
                                               my_mu - true_mu) * dx)))
    assert mu_error_norm < 0.01


def test_instationary_Stokes_control_CN_with_exact_sol():
    mesh_size = 3
    mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size, quadrilateral=True)

    space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_p = FunctionSpace(mesh, "Lagrange", 1)

    n_t = 10
    time_interval = (0.0, 1.0)

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)

        T_f = 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                exp(T_f - t) * 20.0 * X[0] * (X[1]**3),
                exp(T_f - t) * 5.0 * (X[0]**4 - X[1]**4)]),
            "on_boundary")

        return my_bcs

    beta = 10.0**-3

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.
        beta = 10.0**-3

        v_d_help = Function(space)
        v_d_help.interpolate(as_vector([
            4. * beta * X[1] * (2. * (3. * X[0] * X[0] - 1.) * (X[1] * X[1] - 1.) + 3. * (X[0] * X[0] - 1.)**2),  # noqa: E501
            -4. * beta * X[0] * (3. * (X[1] * X[1] - 1.)**2 + 2. * (X[0] * X[0] - 1.) * (3. * X[1] * X[1] - 1.))]))  # noqa: E501

        # desired state
        v_d = Function(space)
        v_d.interpolate(as_vector([
            exp(T_f - t) * (20. * X[0] * X[1]**3 + 2. * beta * (((X[0] * X[0] - 1.)**2) * (X[1] * X[1] - 7.) - 4. * (3. * X[0] * X[0] - 1.) * (X[1] * X[1] - 1.) + 2.)),  # noqa: E501
            exp(T_f - t) * (5. * (X[0]**4 - X[1]**4) - 2. * beta * X[0] * (((X[1] * X[1] - 1.)**2) * (X[0] * X[0] - 7.) - 4. * (X[0] * X[0] - 1.) * (3. * X[1] * X[1] - 1.) - 2.))]))  # noqa: E501

        with v_d.dat.vec as b_v, \
                v_d_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.
        t = 0.0

        v_0 = Function(space)
        v_0.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        # force function
        f_help = Function(space)
        f_help.interpolate(as_vector([
            2. * X[1] * (X[0]**2 - 1.)**2 * (X[1]**2 - 1.),
            -2. * X[0] * (X[0]**2 - 1.) * (X[1]**2 - 1.)**2]))

        # force function
        f = Function(space)
        f.interpolate(as_vector([
            exp(T_f - t) * (-20. * X[0] * X[1]**3 - 2. * X[1] * (X[0] * X[0] - 1.)**2 * (X[1] * X[1] - 1.)),  # noqa: E501
            exp(T_f - t) * (5. * (X[1]**4 - X[0]**4) + 2. * X[0] * (X[0] * X[0] - 1.) * (X[1] * X[1] - 1.)**2)]))  # noqa: E501

        with f.dat.vec as b_v, \
                f_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        return inner(f, test) * dx

    my_control_instationary = Control.Instationary(
        space_v, forw_diff_operator_v, desired_state_v, force_f_v,
        beta=beta, initial_condition=initial_condition_v,
        time_interval=time_interval, CN=True, n_t=n_t,
        bcs_v=my_DirichletBC_t_v)

    lambda_v_bounds = (0.25, 1.5625)
    lambda_p_bounds = (0.25, 2.25)

    my_control_instationary.incompressible_linear_solve(
        ConstantNullspace(), space_p=space_p, lambda_v_bounds=lambda_v_bounds,
        lambda_p_bounds=lambda_p_bounds, print_error=False,
        create_output=False)

    my_v = my_control_instationary._v
    my_p = my_control_instationary._p

    my_zeta = my_control_instationary._zeta
    my_mu = my_control_instationary._mu

    def v_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(T_f - t) * 20. * X[0] * X[1]**3,
            exp(T_f - t) * (5. * X[0]**4 - 5. * X[1]**4)]))

        return true_v

    def p_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        true_p = Function(space)
        true_p.interpolate(
            exp(T_f - t) * (60. * X[0]**2 * X[1] - 20. * X[1]**3))

        return true_p

    def zeta_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        beta = 10.0**-3

        true_zeta = Function(space)
        true_zeta.interpolate(as_vector([
            beta * (exp(T_f - t) - 1.) * 2. * X[1] * (X[0]**2 - 1.)**2 * (X[1]**2 - 1.),  # noqa: E501
            -beta * (exp(T_f - t) - 1.) * 2. * X[0] * (X[0]**2 - 1.) * (X[1]**2 - 1.)**2]))  # noqa: E501

        return true_zeta

    def mu_sol(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        T_f = 1.0

        beta = 10.0**-3

        true_mu = Function(space)
        true_mu.interpolate(
            beta * exp(T_f - t) * 4. * X[0] * X[1])

        return true_mu

    flattened_space_v = tuple(space_v for i in range(n_t))
    mixed_element_v = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_v])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

    flattened_space_p = tuple(space_p for i in range(n_t - 1))
    mixed_element_p = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space_p])
    full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

    true_v = Function(full_space_v, name="true_v")
    true_zeta = Function(full_space_v, name="true_zeta")

    true_p = Function(full_space_p, name="true_p")
    true_mu = Function(full_space_p, name="true_mu")

    t_0 = time_interval[0]
    T_f = time_interval[1]
    tau = (T_f - t_0) / (n_t - 1.0)

    v_test = TestFunction(space_v)
    p_test = TestFunction(space_p)

    true_v_i = v_sol(v_test, t_0)
    true_v.sub(0).assign(true_v_i)

    true_zeta_i = zeta_sol(v_test, t_0)
    true_zeta.sub(0).assign(true_zeta_i)

    for i in range(1, n_t):
        t = t_0 + i * tau
        t_p = t_0 + 0.5 * i * tau

        true_v_i = v_sol(v_test, t)
        true_v.sub(i).assign(true_v_i)

        true_zeta_i = zeta_sol(v_test, t)
        true_zeta.sub(i).assign(true_zeta_i)

        true_p_i = p_sol(p_test, t_p)
        true_p.sub(i - 1).assign(true_p_i)

        true_mu_i = mu_sol(p_test, t_p)
        true_mu.sub(i - 1).assign(true_mu_i)

    v_error_norm = np.sqrt(abs(assemble(inner(my_v - true_v,
                                              my_v - true_v) * dx)))
    assert v_error_norm < 0.01

    zeta_error_norm = np.sqrt(abs(assemble(inner(my_zeta - true_zeta,
                                                 my_zeta - true_zeta) * dx)))
    assert zeta_error_norm < 0.01

    p_error_norm = np.sqrt(abs(assemble(inner(my_p - true_p,
                                              my_p - true_p) * dx)))
    assert p_error_norm < 0.01

    mu_error_norm = np.sqrt(abs(assemble(inner(my_mu - true_mu,
                                               my_mu - true_mu) * dx)))
    assert mu_error_norm < 0.01


def test_instationary_Navier_Stokes_BE():
    # defining the mesh
    mesh_size = 3
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size, 2.0, 2.0)

    space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_p = FunctionSpace(mesh, "Lagrange", 1)

    n_t = 10
    time_interval = (0.0, 2.0)

    def my_DirichletBC_t_v(space_v, t):
        if t < 1.0:
            my_bcs = [DirichletBC(space_v, Constant((t, 0.0)), (4,)),
                      DirichletBC(space_v, 0.0, (1, 2, 3))]
        else:
            my_bcs = [DirichletBC(space_v, Constant((1.0, 0.0)), (4,)),
                      DirichletBC(space_v, 0.0, (1, 2, 3))]

        return my_bcs

    beta = 10.0**-3

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        nu = 1.0 / 50.0
        return nu * inner(grad(trial), grad(test)) * dx + inner(dot(u, grad(trial)), test) * dx  # noqa: E501

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        a = (100.0 / 49.0) ** 2
        b = (100.0 / 99.0) ** 2

        c_1 = 1.0 - sqrt(a * ((x - 0.5) ** 2)
                         + b * (y ** 2))
        c_2 = 1.0 - sqrt(a * ((x + 0.5) ** 2)
                         + b * (y ** 2))
        v_d = Function(space, name="v_d")
        v_d.interpolate(
            ufl.conditional(
                c_1 >= 0.0,
                c_1 * cos(pi * t / 2.0) * as_vector((b * y, -a * (x - 0.5))),
                ufl.conditional(
                    c_2 >= 0.0,
                    c_2 * cos(pi * t / 2.0) * as_vector((-b * y, a * (x + 0.5))),  # noqa: E501
                    as_vector((0.0, 0.0)))),
        )

        return inner(v_d, test) * dx, v_d

    def initial_condition_v(test):
        space = test.function_space()

        v_0 = Function(space)
        v_0.interpolate(as_vector([0.0, 0.0]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()

        # force function
        f = Function(space)
        f.interpolate(as_vector([0.0, 0.0]))

        return inner(f, test) * dx

    my_control_instationary = Control.Instationary(
        space_v, forw_diff_operator_v, desired_state_v, force_f_v,
        beta=beta, initial_condition=initial_condition_v,
        time_interval=time_interval, CN=False, n_t=n_t,
        bcs_v=my_DirichletBC_t_v)

    lambda_v_bounds = (0.3924, 2.0598)
    lambda_p_bounds = (0.5, 2.0)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 100,
                         "relative_tolerance": 1.0e-8,
                         "absolute_tolerance": 0.0,
                         "monitor_convergence": True}

    my_control_instationary.incompressible_non_linear_solve(
        ConstantNullspace(), space_p=space_p,
        solver_parameters=solver_parameters,
        lambda_v_bounds=lambda_v_bounds, lambda_p_bounds=lambda_p_bounds,
        relative_non_linear_tol=10**-5, max_non_linear_iter=10,
        print_error_linear=False, create_output=False)


def test_instationary_Navier_Stokes_CN():
    # defining the mesh
    mesh_size = 3
    mesh = RectangleMesh(2 ** mesh_size, 2 ** mesh_size, 2.0, 2.0)

    space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_p = FunctionSpace(mesh, "Lagrange", 1)

    n_t = 10
    time_interval = (0.0, 2.0)

    def my_DirichletBC_t_v(space_v, t):
        if t < 1.0:
            my_bcs = [DirichletBC(space_v, Constant((t, 0.0)), (4,)),
                      DirichletBC(space_v, 0.0, (1, 2, 3))]
        else:
            my_bcs = [DirichletBC(space_v, Constant((1.0, 0.0)), (4,)),
                      DirichletBC(space_v, 0.0, (1, 2, 3))]

        return my_bcs

    beta = 10.0**-3

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        nu = 1.0 / 50.0
        return nu * inner(grad(trial), grad(test)) * dx + inner(dot(u, grad(trial)), test) * dx  # noqa: E501

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        a = (100.0 / 49.0) ** 2
        b = (100.0 / 99.0) ** 2

        c_1 = 1.0 - sqrt(a * ((x - 0.5) ** 2)
                         + b * (y ** 2))
        c_2 = 1.0 - sqrt(a * ((x + 0.5) ** 2)
                         + b * (y ** 2))
        v_d = Function(space, name="v_d")
        v_d.interpolate(
            ufl.conditional(
                c_1 >= 0.0,
                c_1 * cos(pi * t / 2.0) * as_vector((b * y, -a * (x - 0.5))),
                ufl.conditional(
                    c_2 >= 0.0,
                    c_2 * cos(pi * t / 2.0) * as_vector((-b * y, a * (x + 0.5))),  # noqa: E501
                    as_vector((0.0, 0.0)))),
        )

        return inner(v_d, test) * dx, v_d

    def initial_condition_v(test):
        space = test.function_space()

        v_0 = Function(space)
        v_0.interpolate(as_vector([0.0, 0.0]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()

        # force function
        f = Function(space)
        f.interpolate(as_vector([0.0, 0.0]))

        return inner(f, test) * dx

    my_control_instationary = Control.Instationary(
        space_v, forw_diff_operator_v, desired_state_v, force_f_v,
        beta=beta, initial_condition=initial_condition_v,
        time_interval=time_interval, CN=True, n_t=n_t,
        bcs_v=my_DirichletBC_t_v)

    lambda_v_bounds = (0.3924, 2.0598)
    lambda_p_bounds = (0.5, 2.0)

    solver_parameters = {"linear_solver": "fgmres",
                         "fgmres_restart": 10,
                         "maximum_iterations": 100,
                         "relative_tolerance": 1.0e-8,
                         "absolute_tolerance": 0.0,
                         "monitor_convergence": True}

    my_control_instationary.incompressible_non_linear_solve(
        ConstantNullspace(), space_p=space_p,
        solver_parameters=solver_parameters,
        lambda_v_bounds=lambda_v_bounds, lambda_p_bounds=lambda_p_bounds,
        relative_non_linear_tol=10**-5, max_non_linear_iter=10,
        print_error_linear=False, create_output=False)
