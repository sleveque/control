from firedrake import *
from control.control import Stationary

from time import perf_counter

def forw_diff_operator(trial, test, v):
    return inner(grad(trial), grad(test)) * dx

def desired_state(test):
    space = test.function_space()
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)
    x = X[0] - 1.0
    y = X[1] - 1.0

    v_d = Function(space, name="v_d")
    v_d.interpolate(cos(0.5 * pi * x) * cos(0.5 * pi * y) + 1.0)

    return inner(v_d, test) * dx, v_d

for beta_exp in range(0, 7):
    for mesh_size in range(4, 10):
        mesh = RectangleMesh(2**mesh_size, 2**mesh_size, 2.0, 2.0)
        space_0 = FunctionSpace(mesh, "Lagrange", 1)

        bc = DirichletBC(space_0, 1.0, "on_boundary")

        control_stationary = Stationary(
            space_0, forw_diff_operator, desired_state=desired_state,
            beta=10.0**-beta_exp, bcs_v=bc)

        print('Beta:', 10.0**-beta_exp)
        print('Mesh-size:', mesh_size)
        t0 = perf_counter()
        control_stationary.linear_solve()
        t1 = perf_counter()
        print('Elapsed time:', t1 - t0)
        print('\n')
