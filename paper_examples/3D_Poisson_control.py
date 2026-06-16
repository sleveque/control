from firedrake import *
from control.control import Stationary
import time

beta_range = [1.0, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12]
l_range = [1, 2, 3, 4, 5, 6]

def forw_diff_operator(trial, test, v):
    return inner(grad(trial), grad(test)) * dx

def desired_state(test):
    space = test.function_space()
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)
    x = X[0] - 1.0
    y = X[1] - 1.0
    z = X[2] - 1.0

    v_d = Function(space, name="v_d")
    v_d.interpolate(
        cos(0.5 * pi * x) * cos(0.5 * pi * y) * cos(0.5 * pi * z) + 1.0)

    return inner(v_d, test) * dx, v_d

for beta_i in beta_range:
    print(f"{beta_i=}")
    for l in l_range:
        print(f"{l=}")
        mesh = BoxMesh(2**l, 2**l, 2**l, 2.0, 2.0, 2.0)
        space_0 = FunctionSpace(mesh, "Lagrange", 1)

        bc = DirichletBC(space_0, 1.0, "on_boundary")

        control_stationary = Stationary(
            space_0, forw_diff_operator, desired_state=desired_state,
            beta=beta_i, bcs_v=bc)

        start = time.time()
        control_stationary.linear_solve()
        end = time.time()
        total_time = end - start
        print(f'Elapsed time: {total_time:.16e}')
        print()

        # add comparison with other software, employing LU?
