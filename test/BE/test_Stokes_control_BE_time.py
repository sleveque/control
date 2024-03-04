#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *

from preconditioner.preconditioner import *

from control.control import *

import petsc4py.PETSc as PETSc
import mpi4py.MPI as MPI
import numpy as np
import ufl
import pytest


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


def test_MMS_instationary_Stokes_control_BE_convergence_time():
    degree_range = (2, 4)
    p_range = (2, 5)
    beta = 10.0 ** -3
    t_f = 2.0
    time_interval = (0.0, t_f)

    def v_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t_f)

        v = as_vector([
            exp(a - t) * x * y**3,
            (1. / 4.) * exp(a - t) * (x**4 - y**4)])

        return v

    def p_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t_f)

        p = exp(a - t) * (3. * x**2 * y - y**3)

        return p

    def zeta_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t_f)

        zeta = as_vector([
            beta * (exp(a - t) - 1.) * 2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -beta * (exp(a - t) - 1.) * 2. * x * (x**2 - 1.) * (y**2 - 1.)**2])

        return zeta

    def mu_sol(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0
        a = Constant(t_f)

        mu = beta * exp(a - t) * 4. * x * y

        return mu

    def my_DirichletBC_t_v(space_v, t):
        mesh = space_v.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t_f)

        my_bcs = DirichletBC(
            space_v,
            as_vector([
                exp(a - t) * x * (y**3),
                (1. / 4.) * exp(a - t) * (x**4 - y**4)]),
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
        a = Constant(t_f)

        v_d_help = Function(space, name="v_d_help")
        v_d_help.interpolate(as_vector([
            4. * beta * y * (2. * (3. * x * x - 1.) * (y * y - 1.) + 3. * (x * x - 1.)**2),  # noqa: E501
            -4. * beta * x * (3. * (y * y - 1.)**2 + 2. * (x * x - 1.) * (3. * y * y - 1.))]))  # noqa: E501

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(as_vector([
            exp(a - t) * (x * y**3 + 2. * beta * y * (((x * x - 1.)**2) * (y * y - 7.) - 4. * (3. * x * x - 1.) * (y * y - 1.) + 2.)),  # noqa: E501
            exp(a - t) * ((1. / 4.) * (x**4 - y**4) - 2. * beta * x * (((y * y - 1.)**2) * (x * x - 7.) - 4. * (x * x - 1.) * (3. * y * y - 1.) - 2.))]))  # noqa: E501

        with v_d.dat.vec as b_v, \
                v_d_help.dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

        true_v = Function(space, name="true_v")
        true_v.interpolate(as_vector([
            exp(a - t) * x * y**3,
            (1. / 4.) * exp(a - t) * (x**4 - y**4)]))

        return inner(v_d, test) * dx, true_v

    def initial_condition_v(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t_f)

        v_0 = Function(space)
        v_0.interpolate(as_vector([
            exp(a) * x * y**3,
            (1. / 4.) * exp(a) * (x**4 - y**4)]))

        return v_0

    def force_f_v(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0
        a = Constant(t_f)

        # force function
        f_help = Function(space, name="f_help")
        f_help.interpolate(as_vector([
            2. * y * (x**2 - 1.)**2 * (y**2 - 1.),
            -2. * x * (x**2 - 1.) * (y**2 - 1.)**2]))

        # force function
        f = Function(space, name="f")
        f.interpolate(as_vector([
            exp(a - t) * (-x * y**3 - 2. * y * (x * x - 1.)**2 * (y * y - 1.)),  # noqa: E501
            exp(a - t) * ((1. / 4.) * (y**4 - x**4) + 2. * x * (x * x - 1.) * (y * y - 1.)**2)]))  # noqa: E501

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
                time_interval=time_interval, CN=False, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            if degree == 2:
                lambda_v_bounds = (0.3924, 2.0598)
                lambda_p_bounds = (0.5, 2.0)
            else:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 400,
                                 "relative_tolerance": 1.0e-7,
                                 "absolute_tolerance": 1.0e-7,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                print_error=False, create_output=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            full_space_v = MixedFunctionSpace(flattened_space_v)

            flattened_space_p = tuple(space_p for i in range(n_t - 1))
            full_space_p = MixedFunctionSpace(flattened_space_p)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            full_space_v_ref = MixedFunctionSpace(flattened_space_v_ref)

            flattened_space_p_ref = tuple(space_p_ref for i in range(n_t - 1))
            full_space_p_ref = MixedFunctionSpace(flattened_space_p_ref)

            my_v = Function(full_space_v)
            my_p = Function(full_space_p)

            my_zeta = Function(full_space_v)
            my_mu = Function(full_space_p)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            for i in range(n_t - 1):
                my_p.sub(i).assign(my_control_instationary._p.sub(i + 1))
                my_mu.sub(i).assign(my_control_instationary._mu.sub(i))

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

                v_ref.sub(i).interpolate(v_sol(*X, Constant(t)))

                zeta_ref.sub(i).interpolate(zeta_sol(*X, Constant(t)))

                true_p_i_ref.interpolate(p_sol(*X, Constant(t + tau)))
                mean = assemble(true_p_i_ref * dx)
                with true_p_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                p_ref.sub(i).assign(true_p_i_ref)

                p_help.assign(my_p.sub(i))
                mean = assemble(p_help * dx)
                with p_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_p.sub(i).assign(p_help)

                true_mu_i_ref.interpolate(mu_sol(*X, Constant(t)))
                mean = assemble(true_mu_i_ref * dx)
                with true_mu_i_ref.dat.vec as b_p:
                    b_p.shift(-mean)
                mu_ref.sub(i).assign(true_mu_i_ref)

                mu_help.assign(my_mu.sub(i))
                mean = assemble(mu_help * dx)
                with mu_help.dat.vec as b_p:
                    b_p.shift(-mean)
                my_mu.sub(i).assign(mu_help)

            v_ref.sub(n_t - 1).interpolate(v_sol(*X, Constant(t_f)))

            zeta_ref.sub(n_t - 1).interpolate(zeta_sol(*X, Constant(t_f)))

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
