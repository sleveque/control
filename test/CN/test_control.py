#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *

from preconditioner.preconditioner import *

from control.control import *

from tlm_adjoint.firedrake import (
    DirichletBCApplication, Functional, compute_gradient, minimize_scipy,
    reset_manager, start_manager, stop_manager)

import petsc4py.PETSc as PETSc
import mpi4py.MPI as MPI
import numpy as np
import ufl
import pytest


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


def test_MMS_instationary_heat_control_CN_convergence_time():
    degree_range = (1, 3)
    p_range = (2, 6)
    beta = 1.0
    t_f = 2.0

    def ref_sol_v(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta)) * exp(Constant(t))

        return 1.0 + (c_1 + c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def ref_sol_zeta(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = exp(Constant(t_f))
        c_2 = - exp(Constant(t))

        return (c_1 + c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def forw_diff_operator(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t)

        # desired state
        v_d = Function(space, name="v_d")
        v_sol = Function(space, name="v_sol")

        c_1 = (2.0 / (pi * pi * beta) + 0.5 * pi * pi) * exp(Constant(t_f))
        c_2 = (1.0 - 2.0 / ((2.0 + pi * pi) * beta) - 0.5 * pi * pi) * exp(a)
        c = c_1 + c_2

        v_d.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta)) * exp(a)

        v_sol.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        return inner(v_d, test) * dx, v_sol

    def initial_condition(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta))
        c = c_1 + c_2

        v_0 = Function(space)
        v_0.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        return v_0

    def force_f(test, t):
        space = test.function_space()

        # force function
        f = Function(space)

        f.interpolate(Constant(0.0))

        return inner(f, test) * dx

    def my_DirichletBC_t(space_0, t):
        my_DirichletBC = DirichletBC(space_0, 1.0, "on_boundary")
        return my_DirichletBC

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 300
            n_t = 2 ** p
            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)
            space_0 = FunctionSpace(mesh, "Lagrange", degree)
            space_0_ref = FunctionSpace(mesh, "Lagrange", degree + 2)

            my_control_instationary = Control.Instationary(
                space_0, forw_diff_operator, desired_state, force_f,
                beta=beta, n_t=n_t,
                initial_condition=initial_condition,
                time_interval=(0.0, t_f), bcs_v=my_DirichletBC_t)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-10,
                                 "absolute_tolerance": 1.0e-10,
                                 "monitor_convergence": False}

            my_control_instationary.linear_solve(
                solver_parameters=solver_parameters,
                print_error=False, create_output=False, plots=False)

            flattened_space_v = tuple(space_0 for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_0.mesh(), mixed_element_v)

            flattened_space_v_ref = tuple(space_0_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(
                space_0_ref.mesh(), mixed_element_v_ref)

            my_v = Function(full_space_v)
            my_zeta = Function(full_space_v)
            v_ref = Function(full_space_v_ref)
            zeta_ref = Function(full_space_v_ref)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            del my_control_instationary
            PETSc.garbage_cleanup(space_0.mesh().comm)

            tau = t_f / (n_t - 1.0)

            for i in range(n_t):
                t = i * tau

                v_ref.sub(i).interpolate(ref_sol_v(*X, t))
                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, t))

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")


def test_MMS_instationary_convection_diffusion_control_CN_convergence_time():
    degree_range = (1, 3)
    p_range = (2, 6)
    beta = 1.0
    t_f = 2.0

    def ref_sol_v(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta)) * exp(Constant(t))

        return 1.0 + (c_1 + c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def ref_sol_zeta(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = exp(Constant(t_f))
        c_2 = - exp(Constant(t))

        return (c_1 + c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def forw_diff_operator(trial, test, u, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t)
        wind = as_vector([cos(0.5 * pi * a) * 2. * y * (1. - x * x),
                          -cos(0.5 * pi * a) * 2. * x * (1. - y * y)])
        # spatial differential for the forward problem
        return (
            inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), wind), test) * dx)

    def desired_state(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t)
        wind = as_vector([cos(0.5 * pi * a) * 2. * y * (1. - x * x),
                          -cos(0.5 * pi * a) * 2. * x * (1. - y * y)])
        zeta = ref_sol_zeta(*X, t)
        grad_zeta = grad(zeta)
        convection = grad_zeta[(0)] * wind[(0)] + grad_zeta[(1)] * wind[(1)]

        # desired state
        v_d = Function(space, name="v_d")
        v_sol = Function(space, name="v_sol")

        c_1 = (2.0 / (pi * pi * beta) + 0.5 * pi * pi) * exp(Constant(t_f))
        c_2 = (1.0 - 2.0 / ((2.0 + pi * pi) * beta) - 0.5 * pi * pi) * exp(a)
        c = c_1 + c_2

        v_d.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y)
            - convection)

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta)) * exp(a)
        c = c_1 + c_2

        v_sol.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        return inner(v_d, test) * dx, v_sol

    def initial_condition(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        c_1 = (2.0 / (pi * pi * beta)) * exp(Constant(t_f))
        c_2 = - (2.0 / ((2.0 + pi * pi) * beta))
        c = c_1 + c_2

        v_0 = Function(space)
        v_0.interpolate(
            Constant(1.0) + c * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        return v_0

    def force_f(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t)
        wind = as_vector([cos(0.5 * pi * a) * 2. * y * (1. - x * x),
                          -cos(0.5 * pi * a) * 2. * x * (1. - y * y)])
        v = ref_sol_v(*X, t)
        grad_v = grad(v)
        convection = grad_v[(0)] * wind[(0)] + grad_v[(1)] * wind[(1)]

        # force function
        f = Function(space)

        f.interpolate(convection)

        return inner(f, test) * dx

    def my_DirichletBC_t(space_0, t):
        my_DirichletBC = DirichletBC(space_0, 1.0, "on_boundary")
        return my_DirichletBC

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 300
            n_t = 2 ** p
            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)
            space_0 = FunctionSpace(mesh, "Lagrange", degree)
            space_0_ref = FunctionSpace(mesh, "Lagrange", degree + 2)

            my_control_instationary = Control.Instationary(
                space_0, forw_diff_operator, desired_state, force_f,
                beta=beta, n_t=n_t,
                initial_condition=initial_condition,
                time_interval=(0.0, t_f), bcs_v=my_DirichletBC_t)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-10,
                                 "absolute_tolerance": 1.0e-10,
                                 "monitor_convergence": False}

            my_control_instationary.linear_solve(
                solver_parameters=solver_parameters,
                print_error=True, create_output=False, plots=False)

            flattened_space_v = tuple(space_0 for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_0.mesh(), mixed_element_v)

            flattened_space_v_ref = tuple(space_0_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(
                space_0_ref.mesh(), mixed_element_v_ref)

            my_v = Function(full_space_v)
            my_zeta = Function(full_space_v)
            v_ref = Function(full_space_v_ref)
            zeta_ref = Function(full_space_v_ref)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            del my_control_instationary
            PETSc.garbage_cleanup(space_0.mesh().comm)

            tau = t_f / (n_t - 1.0)

            for i in range(n_t):
                t = i * tau

                v_ref.sub(i).interpolate(ref_sol_v(*X, t))
                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, t))

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")


def test_MMS_instationary_Stokes_control_CN_convergence_FE():
    degree_range = (2, 4)
    p_range = (2, 6)
    beta = 10.0 ** -1
    t_f = 2.0
    time_interval = (0.0, t_f)

    def v_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        v = as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)])

        return v

    def p_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        p = exp(t_f - t) * (3. * x**2 * y - y**3)

        return p

    def zeta_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t)
        b = Constant(t_f)

        zeta = as_vector([
            beta * (exp(b - a) - 1.) * 2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -beta * (exp(b - a) - 1.) * 2. * x * (x**2 - 1.) * (y**2 - 1.)**2])

        return zeta

    def mu_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        mu = beta * exp(t_f - t) * 4. * x * y

        return mu

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                exp(t_f - t) * x * (y**3),
                (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]),
            "on_boundary")

        return my_bcs

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        v_d_help = Function(space)
        v_d_help.interpolate(as_vector([
            4. * beta * y * (2. * (3. * x * x - 1.) * (y * y - 1.) + 3. * (x * x - 1.)**2),  # noqa: E501
            -4. * beta * x * (3. * (y * y - 1.)**2 + 2. * (x * x - 1.) * (3. * y * y - 1.))]))  # noqa: E501

        # desired state
        v_d = Function(space)
        v_d.interpolate(as_vector([
            exp(t_f - t) * (x * y**3 + 2. * beta * y * (((x * x - 1.)**2) * (y * y - 7.) - 4. * (3. * x * x - 1.) * (y * y - 1.) + 2.)),  # noqa: E501
            exp(t_f - t) * ((1. / 4.) * (x**4 - y**4) - 2. * beta * x * (((y * y - 1.)**2) * (x * x - 7.) - 4. * (x * x - 1.) * (3. * y * y - 1.) - 2.))]))  # noqa: E501

        with v_d.dat.vec as b_v, \
                v_d_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]))

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        t = 0.0

        v_0 = Function(space)
        v_0.interpolate(as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        # force function
        f_help = Function(space)
        f_help.interpolate(as_vector([
            2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -2. * x * (x**2 - 1.) * (y**2 - 1.)**2]))

        # force function
        f = Function(space)
        f.interpolate(as_vector([
            exp(t_f - t) * (-x * y**3 - 2. * y * (x * x - 1.)**2 * (y * y - 1.)),  # noqa: E501
            exp(t_f - t) * ((1. / 4.) * (y**4 - x**4) + 2. * x * (x * x - 1.) * (y * y - 1.)**2)]))  # noqa: E501

        with f.dat.vec as b_v, \
                f_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        return inner(f, test) * dx

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        p_error_norms = []
        mu_error_norms = []
        for p in range(*p_range):
            N = 2 ** p
            n_t = 100

            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)

            space_v = VectorFunctionSpace(mesh, "Lagrange", degree)
            space_p = FunctionSpace(mesh, "Lagrange", degree - 1)

            space_v_ref = VectorFunctionSpace(mesh, "Lagrange", degree + 2)
            space_p_ref = FunctionSpace(mesh, "Lagrange", degree + 1)

            my_control_instationary = Control.Instationary(
                space_v, forw_diff_operator_v, desired_state_v, force_f_v,
                beta=beta, initial_condition=initial_condition_v,
                time_interval=time_interval, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            if degree == 2:
                lambda_v_bounds = (0.3924, 2.0598)
                lambda_p_bounds = (0.5, 2.0)
            else:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-12,
                                 "absolute_tolerance": 1.0e-12,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                print_error=False, create_output=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            flattened_space_p = tuple(space_p for i in range(n_t - 1))
            mixed_element_p = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_p])
            full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(space_v_ref.mesh(), mixed_element_v_ref)  # noqa: E501

            flattened_space_p_ref = tuple(space_p_ref for i in range(n_t - 1))
            mixed_element_p_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_p_ref])
            full_space_p_ref = FunctionSpace(space_p_ref.mesh(), mixed_element_p_ref)  # noqa: E501

            my_v = Function(full_space_v)
            my_p = Function(full_space_p)

            my_zeta = Function(full_space_v)
            my_mu = Function(full_space_p)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            my_p.assign(my_control_instationary._p)
            my_mu.assign(my_control_instationary._mu)

            v_ref = Function(full_space_v_ref, name="v_ref")
            zeta_ref = Function(full_space_v_ref, name="zeta_ref")

            p_ref = Function(full_space_p_ref, name="p_ref")
            mu_ref = Function(full_space_p_ref, name="mu_ref")

            tau = t_f / (n_t - 1.0)

            p_help = Function(space_p)
            mu_help = Function(space_p)

            true_p_i_ref = Function(space_p_ref)
            true_mu_i_ref = Function(space_p_ref)

            for i in range(n_t - 1):
                t = i * tau

                v_ref.sub(i).interpolate(v_sol(*X, t))

                zeta_ref.sub(i).interpolate(zeta_sol(*X, t))

                true_p_i_ref.interpolate(p_sol(*X, t + tau))
                mean = assemble(true_p_i_ref * dx)
                with true_p_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                p_ref.sub(i).assign(true_p_i_ref)

                p_help.assign(my_p.sub(i))
                mean = assemble(p_help * dx)
                with p_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_p.sub(i).assign(p_help)

                true_mu_i_ref.interpolate(mu_sol(*X, t))
                mean = assemble(true_mu_i_ref * dx)
                with true_mu_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                mu_ref.sub(i).assign(true_mu_i_ref)

                mu_help.assign(my_mu.sub(i))
                mean = assemble(mu_help * dx)
                with mu_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_mu.sub(i).assign(mu_help)

            v_ref.sub(n_t - 1).interpolate(v_sol(*X, t_f))

            zeta_ref.sub(n_t - 1).interpolate(zeta_sol(*X, t_f))

            del my_control_instationary
            PETSc.garbage_cleanup(space_v.mesh().comm)

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {N=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {N=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

            p_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_p - p_ref, my_p - p_ref) * dx)))
            print(f"{degree=} {p=} {N=} {p_error_norm=}")
            p_error_norms.append(p_error_norm)

            mu_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_mu - mu_ref, my_mu - mu_ref) * dx)))
            print(f"{degree=} {p=} {N=} {mu_error_norm=}")
            mu_error_norms.append(mu_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")

        p_error_norms = np.array(p_error_norms)
        p_orders = np.log(p_error_norms[:-1] / p_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {p_orders=}")

        mu_error_norms = np.array(mu_error_norms)
        mu_orders = np.log(mu_error_norms[:-1] / mu_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {mu_orders=}")


def test_MMS_instationary_Stokes_control_CN_convergence_time():
    degree_range = (2, 4)
    p_range = (2, 6)
    beta = 10.0 ** -1
    t_f = 2.0
    time_interval = (0.0, t_f)

    def v_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        v = as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)])

        return v

    def p_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        p = exp(t_f - t) * (3. * x**2 * y - y**3)

        return p

    def zeta_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t)
        b = Constant(t_f)

        zeta = as_vector([
            beta * (exp(b - a) - 1.) * 2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -beta * (exp(b - a) - 1.) * 2. * x * (x**2 - 1.) * (y**2 - 1.)**2])

        return zeta

    def mu_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        mu = beta * exp(t_f - t) * 4. * x * y

        return mu

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                exp(t_f - t) * x * (y**3),
                (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]),
            "on_boundary")

        return my_bcs

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        v_d_help = Function(space)
        v_d_help.interpolate(as_vector([
            4. * beta * y * (2. * (3. * x * x - 1.) * (y * y - 1.) + 3. * (x * x - 1.)**2),  # noqa: E501
            -4. * beta * x * (3. * (y * y - 1.)**2 + 2. * (x * x - 1.) * (3. * y * y - 1.))]))  # noqa: E501

        # desired state
        v_d = Function(space)
        v_d.interpolate(as_vector([
            exp(t_f - t) * (x * y**3 + 2. * beta * y * (((x * x - 1.)**2) * (y * y - 7.) - 4. * (3. * x * x - 1.) * (y * y - 1.) + 2.)),  # noqa: E501
            exp(t_f - t) * ((1. / 4.) * (x**4 - y**4) - 2. * beta * x * (((y * y - 1.)**2) * (x * x - 7.) - 4. * (x * x - 1.) * (3. * y * y - 1.) - 2.))]))  # noqa: E501

        with v_d.dat.vec as b_v, \
                v_d_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        true_v = Function(space)
        true_v.interpolate(as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]))

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        t = 0.0

        v_0 = Function(space)
        v_0.interpolate(as_vector([
            exp(t_f - t) * x * y**3,
            (1. / 4.) * exp(t_f - t) * (x**4 - y**4)]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        # force function
        f_help = Function(space)
        f_help.interpolate(as_vector([
            2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -2. * x * (x**2 - 1.) * (y**2 - 1.)**2]))

        # force function
        f = Function(space)
        f.interpolate(as_vector([
            exp(t_f - t) * (-x * y**3 - 2. * y * (x * x - 1.)**2 * (y * y - 1.)),  # noqa: E501
            exp(t_f - t) * ((1. / 4.) * (y**4 - x**4) + 2. * x * (x * x - 1.) * (y * y - 1.)**2)]))  # noqa: E501

        with f.dat.vec as b_v, \
                f_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        return inner(f, test) * dx

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        p_error_norms = []
        mu_error_norms = []
        for p in range(*p_range):
            N = 100
            n_t = 2 ** p

            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)

            space_v = VectorFunctionSpace(mesh, "Lagrange", degree)
            space_p = FunctionSpace(mesh, "Lagrange", degree - 1)

            space_v_ref = VectorFunctionSpace(mesh, "Lagrange", degree + 2)
            space_p_ref = FunctionSpace(mesh, "Lagrange", degree + 1)

            my_control_instationary = Control.Instationary(
                space_v, forw_diff_operator_v, desired_state_v, force_f_v,
                beta=beta, initial_condition=initial_condition_v,
                time_interval=time_interval, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            if degree == 2:
                lambda_v_bounds = (0.3924, 2.0598)
                lambda_p_bounds = (0.5, 2.0)
            else:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-12,
                                 "absolute_tolerance": 1.0e-12,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                print_error=False, create_output=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            flattened_space_p = tuple(space_p for i in range(n_t - 1))
            mixed_element_p = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_p])
            full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(space_v_ref.mesh(), mixed_element_v_ref)  # noqa: E501

            flattened_space_p_ref = tuple(space_p_ref for i in range(n_t - 1))
            mixed_element_p_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_p_ref])
            full_space_p_ref = FunctionSpace(space_p_ref.mesh(), mixed_element_p_ref)  # noqa: E501

            my_v = Function(full_space_v)
            my_p = Function(full_space_p)

            my_zeta = Function(full_space_v)
            my_mu = Function(full_space_p)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            my_p.assign(my_control_instationary._p)
            my_mu.assign(my_control_instationary._mu)

            v_ref = Function(full_space_v_ref, name="v_ref")
            zeta_ref = Function(full_space_v_ref, name="zeta_ref")

            p_ref = Function(full_space_p_ref, name="p_ref")
            mu_ref = Function(full_space_p_ref, name="mu_ref")

            tau = t_f / (n_t - 1.0)

            p_help = Function(space_p)
            mu_help = Function(space_p)

            true_p_i_ref = Function(space_p_ref)
            true_mu_i_ref = Function(space_p_ref)

            for i in range(n_t - 1):
                t = i * tau

                v_ref.sub(i).interpolate(v_sol(*X, t))

                zeta_ref.sub(i).interpolate(zeta_sol(*X, t))

                true_p_i_ref.interpolate(p_sol(*X, t + tau))
                mean = assemble(true_p_i_ref * dx)
                with true_p_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                p_ref.sub(i).assign(true_p_i_ref)

                p_help.assign(my_p.sub(i))
                mean = assemble(p_help * dx)
                with p_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_p.sub(i).assign(p_help)

                true_mu_i_ref.interpolate(mu_sol(*X, t))
                mean = assemble(true_mu_i_ref * dx)
                with true_mu_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                mu_ref.sub(i).assign(true_mu_i_ref)

                mu_help.assign(my_mu.sub(i))
                mean = assemble(mu_help * dx)
                with mu_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_mu.sub(i).assign(mu_help)

            v_ref.sub(n_t - 1).interpolate(v_sol(*X, t_f))

            zeta_ref.sub(n_t - 1).interpolate(zeta_sol(*X, t_f))

            del my_control_instationary
            PETSc.garbage_cleanup(space_v.mesh().comm)

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

            p_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_p - p_ref, my_p - p_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {p_error_norm=}")
            p_error_norms.append(p_error_norm)

            mu_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_mu - mu_ref, my_mu - mu_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {mu_error_norm=}")
            mu_error_norms.append(mu_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")

        p_error_norms = np.array(p_error_norms)
        p_orders = np.log(p_error_norms[:-1] / p_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {p_orders=}")

        mu_error_norms = np.array(mu_error_norms)
        mu_orders = np.log(mu_error_norms[:-1] / mu_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {mu_orders=}")


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
        nu = 1.0 / 100.0
        return (
            nu * inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), u), test) * dx)

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
                         "monitor_convergence": False}

    my_control_instationary.incompressible_non_linear_solve(
        ConstantNullspace(), space_p=space_p,
        solver_parameters=solver_parameters,
        lambda_v_bounds=lambda_v_bounds, lambda_p_bounds=lambda_p_bounds,
        relative_non_linear_tol=10**-5, max_non_linear_iter=10,
        print_error_linear=False, create_output=False)

    del my_control_instationary
    PETSc.garbage_cleanup(space_v.mesh().comm)


def test_MMS_instationary_Navier_Stokes_control_CN_convergence_FE():
    degree_range = (2, 4)
    p_range = (2, 6)
    beta = 10.0 ** -3
    t_f = 2.0
    time_interval = (0.0, t_f)
    nu = 1.0 / 50.0

    def ref_sol_v(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        v_xy = as_vector([x * (y ** 3), (1. / 4.) * (x ** 4 - y ** 4)])

        v = cos(pi * Constant(t) / 2.0) * v_xy

        return v, v_xy

    def ref_sol_zeta(x_1, x_2, t):
        zeta = as_vector([0.0, 0.0])  # noqa: E501

        return zeta

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)

        x = X[0] - 1.0
        y = X[1] - 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                cos(pi * t / 2.0) * x * (y ** 3),
                cos(pi * t / 2.0) * (1. / 4.) * (x ** 4 - y ** 4)]),
            "on_boundary")

        return my_bcs

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return (
            nu * inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), u), test) * dx)

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v, v_xy = ref_sol_v(*X, t)

        # desired state
        v_d = Function(space)
        v_d.interpolate(v)

        true_v = Function(space)
        true_v.interpolate(v)

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        v = as_vector([
            x * (y ** 3),
            (1. / 4.) * (x ** 4 - y ** 4)])

        v_0 = Function(space)
        v_0.interpolate(v)

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v, v_xy = ref_sol_v(*X, t)

        # force function
        f = Function(space)
        f.interpolate(
            - 0.5 * nu * div(grad(v) + ufl.transpose(grad(v)))
            + grad(v) * v
            - 0.5 * pi * sin(pi * Constant(t) / 2.0) * v_xy)

        return inner(f, test) * dx

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 2 ** p
            n_t = 100

            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)

            space_v = VectorFunctionSpace(mesh, "Lagrange", degree)
            space_p = FunctionSpace(mesh, "Lagrange", degree - 1)

            space_v_ref = VectorFunctionSpace(mesh, "Lagrange", degree + 2)

            my_control_instationary = Control.Instationary(
                space_v, forw_diff_operator_v, desired_state_v, force_f_v,
                beta=beta, initial_condition=initial_condition_v,
                time_interval=time_interval, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            if degree == 2:
                lambda_v_bounds = (0.3924, 2.0598)
                lambda_p_bounds = (0.5, 2.0)
            else:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-10,
                                 "absolute_tolerance": 1.0e-10,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_non_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                max_non_linear_iter=10, relative_non_linear_tol=10.0**-9,
                absolute_non_linear_tol=10.0**-9,
                print_error_linear=False, print_error_non_linear=False,
                create_output=False, plots=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(
                space_v_ref.mesh(), mixed_element_v_ref)

            my_v = Function(full_space_v)
            my_zeta = Function(full_space_v)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            v_ref = Function(full_space_v_ref, name="v_ref")
            zeta_ref = Function(full_space_v_ref, name="zeta_ref")

            tau = t_f / (n_t - 1.0)

            for i in range(n_t):
                t = i * tau

                v, v_xy = ref_sol_v(*X, t)

                v_ref.sub(i).interpolate(v)

                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, t))

            del my_control_instationary
            PETSc.garbage_cleanup(space_v.mesh().comm)

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {N=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {N=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")


def test_MMS_instationary_Navier_Stokes_control_CN_convergence_time():
    degree_range = (2, 4)
    p_range = (2, 6)
    beta = 10.0 ** -3
    t_f = 2.0
    time_interval = (0.0, t_f)
    nu = 1.0 / 50.0

    def ref_sol_v(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        v_xy = as_vector([x * (y ** 3), (1. / 4.) * (x ** 4 - y ** 4)])

        v = cos(pi * Constant(t) / 2.0) * v_xy

        return v, v_xy

    def ref_sol_zeta(x_1, x_2, t):
        zeta = as_vector([0.0, 0.0])  # noqa: E501

        return zeta

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)

        x = X[0] - 1.0
        y = X[1] - 1.0

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                cos(pi * t / 2.0) * x * (y ** 3),
                cos(pi * t / 2.0) * (1. / 4.) * (x ** 4 - y ** 4)]),
            "on_boundary")

        return my_bcs

    def forw_diff_operator_v(trial, test, u, t):
        # spatial differential for the forward problem
        return (
            nu * inner(grad(trial), grad(test)) * dx
            + inner(dot(grad(trial), u), test) * dx)

    def desired_state_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v, v_xy = ref_sol_v(*X, t)

        # desired state
        v_d = Function(space)
        v_d.interpolate(v)

        true_v = Function(space)
        true_v.interpolate(v)

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        v = as_vector([
            x * (y ** 3),
            (1. / 4.) * (x ** 4 - y ** 4)])

        v_0 = Function(space)
        v_0.interpolate(v)

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)

        v, v_xy = ref_sol_v(*X, t)

        # force function
        f = Function(space)
        f.interpolate(
            - 0.5 * nu * div(grad(v) + ufl.transpose(grad(v)))
            + grad(v) * v
            - 0.5 * pi * sin(pi * Constant(t) / 2.0) * v_xy)

        return inner(f, test) * dx

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 100
            n_t = 2 ** p

            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)

            space_v = VectorFunctionSpace(mesh, "Lagrange", degree)
            space_p = FunctionSpace(mesh, "Lagrange", degree - 1)

            space_v_ref = VectorFunctionSpace(mesh, "Lagrange", degree + 2)

            my_control_instationary = Control.Instationary(
                space_v, forw_diff_operator_v, desired_state_v, force_f_v,
                beta=beta, initial_condition=initial_condition_v,
                time_interval=time_interval, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            if degree == 2:
                lambda_v_bounds = (0.3924, 2.0598)
                lambda_p_bounds = (0.5, 2.0)
            else:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 500,
                                 "relative_tolerance": 1.0e-10,
                                 "absolute_tolerance": 1.0e-10,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_non_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                max_non_linear_iter=10, relative_non_linear_tol=10.0**-9,
                absolute_non_linear_tol=10.0**-9,
                print_error_linear=False, print_error_non_linear=False,
                create_output=False, plots=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            mixed_element_v_ref = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v_ref])
            full_space_v_ref = FunctionSpace(
                space_v_ref.mesh(), mixed_element_v_ref)

            my_v = Function(full_space_v)
            my_zeta = Function(full_space_v)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            v_ref = Function(full_space_v_ref, name="v_ref")
            zeta_ref = Function(full_space_v_ref, name="zeta_ref")

            tau = t_f / (n_t - 1.0)

            for i in range(n_t):
                t = i * tau

                v, v_xy = ref_sol_v(*X, t)

                v_ref.sub(i).interpolate(v)

                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, t))

            del my_control_instationary
            PETSc.garbage_cleanup(space_v.mesh().comm)

            v_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_v - v_ref, my_v - v_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {v_error_norm=}")
            v_error_norms.append(v_error_norm)

            zeta_error_norm = np.sqrt(tau) * np.sqrt(abs(assemble(
                inner(my_zeta - zeta_ref, my_zeta - zeta_ref) * dx)))
            print(f"{degree=} {p=} {n_t=} {zeta_error_norm=}")
            zeta_error_norms.append(zeta_error_norm)

        v_error_norms = np.array(v_error_norms)
        v_orders = np.log(v_error_norms[:-1] / v_error_norms[1:]) / np.log(2.0)
        print(f"{degree=} {v_orders=}")

        zeta_error_norms = np.array(zeta_error_norms)
        zeta_orders = np.log(zeta_error_norms[:-1] / zeta_error_norms[1:]) / np.log(2.0)  # noqa: E501
        print(f"{degree=} {zeta_orders=}")
