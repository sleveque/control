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


nu_range = [1./100, 1./250, 1./500]

beta_range = (3, 6)

l_range = (2, 8)

def my_DirichletBC_t_v(space_v, t):
    if float(t) < 1.0:
        my_bcs = [DirichletBC(space_v, Constant((t, 0.0)), (4,)),
                  DirichletBC(space_v, 0.0, (1, 2, 3))]
    else:
        my_bcs = [DirichletBC(space_v, Constant((1.0, 0.0)), (4,)),
                  DirichletBC(space_v, 0.0, (1, 2, 3))]

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
    x = X[0] - 1.0
    y = X[1] - 1.0

    a = (100.0 / 49.0) ** 2
    b = (100.0 / 99.0) ** 2

    c_1 = (1.0 - sqrt(a * ((x - 0.5) ** 2) + b * (y ** 2)))
    c_2 = (1.0 - sqrt(a * ((x + 0.5) ** 2) + b * (y ** 2)))

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

    return v_0

def force_f_v(test, t):
    space = test.function_space()

    # force function
    f = Function(space)

    return inner(f, test) * dx

lambda_v_bounds = (0.3924, 2.0598)
lambda_p_bounds = (0.5, 2.0)

solver_parameters = {"linear_solver": "fgmres",
                     "fgmres_restart": 10,
                     "maximum_iterations": 100,
                     "relative_tolerance": 1.0e-6,
                     "absolute_tolerance": 0.0,
                     "monitor_convergence": True}

time_interval = (0.0, 2.0)

for nu in nu_range:
    print(f"{nu=}")
    for k in range(*beta_range):
        beta = 10.0**-k
        print(f"{beta=}")

        for l in range(*l_range):
            # defining the mesh
            N = 2**l
            n_t = N

            mesh = RectangleMesh(N, N, 2.0, 2.0)

            space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
            space_p = FunctionSpace(mesh, "Lagrange", 1)

            print(f"{l=}, {N=}")

            my_control_instationary = Control.Instationary(
                space_v, forw_diff_operator_v, desired_state_v, force_f_v,
                beta=beta, initial_condition=initial_condition_v,
                time_interval=time_interval, CN=True, n_t=n_t,
                bcs_v=my_DirichletBC_t_v)

            my_control_instationary.incompressible_non_linear_solve(
                ConstantNullspace(), space_p=space_p,
                solver_parameters=solver_parameters,
                lambda_v_bounds=lambda_v_bounds, lambda_p_bounds=lambda_p_bounds,
                create_output=False)

            print()

            del my_control_instationary
            PETSc.garbage_cleanup(space_v.mesh().comm)

        print()

    print()
