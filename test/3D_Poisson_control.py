from firedrake import *
from control.control import *

mesh = BoxMesh(10, 10, 10, 2.0, 2.0, 2.0)
space_0 = FunctionSpace(mesh, "Lagrange", 1)


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


bc = DirichletBC(space_0, 1.0, "on_boundary")

control_stationary = Stationary(
    space_0, forw_diff_operator, desired_state=desired_state,
    beta=1.0e-4, bcs_v=bc)

control_stationary.linear_solve()
