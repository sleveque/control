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


def test_MMS_instationary_Navier_Stokes_control_CN_convergence_time():
    degree_range = (3, 5)
    p_range = (0, 4)
    beta = 10.0 ** -3
    t_f = 1.0
    time_interval = (0.0, t_f)
    nu = 1.0 / 50.0

    def ref_sol_v(x_1, x_2, t):
        x = x_1 - 1.0
        y = x_2 - 1.0

        v_xy = as_vector([2. * y * (1. - x * x),
                          -2. * x * (1. - y * y)])

        v = cos(pi * t / 2.0) * v_xy

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
                cos(pi * t / 2.0) * 2. * y * (1. - x * x),
                - cos(pi * t / 2.0) * 2. * x * (1. - y * y)]),
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
            2. * y * (1. - x * x),
            -2. * x * (1. - y * y)])

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
            - nu * div(grad(v))
            + grad(v) * v
            - 0.5 * pi * sin(pi * t / 2.0) * v_xy)

        return inner(f, test) * dx

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 5
            n_t = (2 ** p) * 15

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

            if degree == 3:
                lambda_v_bounds = (0.2867, 2.0093)
                lambda_p_bounds = (0.3924, 2.0598)
            else:
                lambda_v_bounds = (0.3772, 2.0856)
                lambda_p_bounds = (0.2867, 2.0093)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 200,
                                 "relative_tolerance": 1.0e-7,
                                 "absolute_tolerance": 1.0e-7,
                                 "monitor_convergence": False}

            my_control_instationary.incompressible_non_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds,
                lambda_p_bounds=lambda_p_bounds,
                max_non_linear_iter=10, relative_non_linear_tol=10.0**-6,
                absolute_non_linear_tol=10.0**-6,
                print_error_linear=False, print_error_non_linear=False,
                create_output=False, plots=False)

            flattened_space_v = tuple(space_v for i in range(n_t))
            full_space_v = MixedFunctionSpace(flattened_space_v)

            flattened_space_v_ref = tuple(space_v_ref for i in range(n_t))
            full_space_v_ref = MixedFunctionSpace(flattened_space_v_ref)

            my_v = Function(full_space_v)
            my_zeta = Function(full_space_v)

            my_v.assign(my_control_instationary._v)
            my_zeta.assign(my_control_instationary._zeta)

            v_ref = Function(full_space_v_ref, name="v_ref")
            zeta_ref = Function(full_space_v_ref, name="zeta_ref")

            tau = t_f / (n_t - 1.0)

            for i in range(n_t):
                t = i * tau

                v, v_xy = ref_sol_v(*X, Constant(t))

                v_ref.sub(i).interpolate(v)

                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, Constant(t)))

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
