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


def test_MMS_instationary_heat_control_BE_convergence_FE():
    degree_range = (1, 3)
    p_range = (2, 6)
    beta = 1.0
    t_f = 2.0

    def ref_sol_v(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = Constant(t_f)
        c_2 = t

        return 1.0 + (c_1 - c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def ref_sol_zeta(x, y, t):
        X_1 = x - 1.0
        X_2 = y - 1.0

        c_1 = Constant(t_f)
        c_2 = t

        return (c_1 - c_2) * cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def ref_sol_v_space(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        return cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def ref_sol_zeta_space(x, y):
        X_1 = x - 1.0
        X_2 = y - 1.0

        return cos(0.5 * pi * X_1) * cos(0.5 * pi * X_2)

    def forw_diff_operator(trial, test, u, t):
        # spatial differential for the forward problem
        return inner(grad(trial), grad(test)) * dx

    def desired_state(test, t):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        zeta = ref_sol_zeta(*X, t)
        v = ref_sol_v(*X, t)

        zeta_space = ref_sol_zeta_space(*X)
        lapl_zeta = div(grad(zeta))

        # desired state
        v_d = Function(space, name="v_d")
        v_d.interpolate(
            zeta_space - lapl_zeta + v)

        # ture v
        true_v = Function(space, name="true_v")
        true_v.assign(v_d)

        return inner(v_d, test) * dx, true_v

    def initial_condition(test):
        space = test.function_space()
        mesh = space.mesh()
        X = SpatialCoordinate(mesh)
        x = X[0] - 1.0
        y = X[1] - 1.0

        c_1 = Constant(t_f)

        v_0 = Function(space)
        v_0.interpolate(
            1.0 + c_1 * cos(0.5 * pi * x) * cos(0.5 * pi * y))

        return v_0

    def force_f(test, t):
        space = test.function_space()
        X = SpatialCoordinate(mesh)

        v = ref_sol_v(*X, t)
        zeta = ref_sol_zeta(*X, t)

        v_space = ref_sol_v_space(*X)

        lapl_v = div(grad(v))

        # force function
        f = Function(space)

        f.interpolate(- v_space - lapl_v - Constant(1. / beta) * zeta)

        return inner(f, test) * dx

    def my_DirichletBC_t(space_0, t):
        my_DirichletBC = DirichletBC(space_0, 1.0, "on_boundary")
        return my_DirichletBC

    for degree in range(*degree_range):
        v_error_norms = []
        zeta_error_norms = []
        for p in range(*p_range):
            N = 2 ** p
            n_t = 10
            mesh = RectangleMesh(N, N, 2.0, 2.0)
            X = SpatialCoordinate(mesh)
            space_0 = FunctionSpace(mesh, "Lagrange", degree)
            space_0_ref = FunctionSpace(mesh, "Lagrange", degree + 2)

            my_control_instationary = Control.Instationary(
                space_0, forw_diff_operator, desired_state, force_f,
                beta=beta, CN=False, n_t=n_t,
                initial_condition=initial_condition,
                time_interval=(0.0, t_f), bcs_v=my_DirichletBC_t)

            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 200,
                                 "relative_tolerance": 1.0e-10,
                                 "absolute_tolerance": 1.0e-10,
                                 "monitor_convergence": False}

            my_control_instationary.linear_solve(
                solver_parameters=solver_parameters,
                print_error=False, create_output=False, plots=False)

            flattened_space_v = tuple(space_0 for i in range(n_t))
            full_space_v = MixedFunctionSpace(flattened_space_v)

            flattened_space_v_ref = tuple(space_0_ref for i in range(n_t))
            full_space_v_ref = MixedFunctionSpace(flattened_space_v_ref)

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

                v_ref.sub(i).interpolate(ref_sol_v(*X, Constant(t)))
                zeta_ref.sub(i).interpolate(ref_sol_zeta(*X, Constant(t)))

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
