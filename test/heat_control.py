from firedrake import *
from control.control import *

mesh = RectangleMesh(10, 10, 1.0, 1.0, originX=-1.0, originY=-1.0)
space_0 = FunctionSpace(mesh, "Lagrange", 1)


def forw_diff_operator(trial, test, v, t):
    return inner(grad(trial), grad(test)) * dx


def desired_state(test, t):
    space = test.function_space()
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)

    v_d = Function(space, name="v_d")
    v_d.interpolate(t * cos(0.5 * pi * X[0]) * cos(0.5 * pi * X[1]))

    return inner(v_d, test) * dx, v_d


def force_f(test, t):
    space = test.function_space()
    mesh = space.mesh()
    X = SpatialCoordinate(mesh)

    f = Function(space, name="f")
    f.interpolate(cos(0.5 * pi * X[0]) * cos(0.5 * pi * X[1]))

    return inner(f, test) * dx


def bc_t(space_0, t):
    return DirichletBC(space_0, 0.0, "on_boundary")

control_instationary = Instationary(
    space_0, forw_diff_operator, desired_state=desired_state,
    force_function=force_f, bcs_v=bc_t, beta=1.0e-4, n_t=10,
    time_interval=(0.0, 2.0))

control_instationary.linear_solve()
