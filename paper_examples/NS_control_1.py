from firedrake import *
from control.control import *
from control.preconditioner import ConstantNullspace
import ufl
from time import perf_counter


def my_DirichletBC_t_v(space_v, t):
    if float(t) < 1.0:
        my_bcs = [DirichletBC(space_v, Constant((t, 0.0)), (4,)),
                  DirichletBC(space_v, 0.0, (1, 2, 3))]
    else:
        my_bcs = [DirichletBC(space_v, Constant((1.0, 0.0)), (4,)),
                  DirichletBC(space_v, 0.0, (1, 2, 3))]
    return my_bcs


def forw_diff_operator_v(trial, test, u, t):
    return (nu * inner(grad(trial), grad(test)) * dx
        + inner(dot(u, grad(trial)), test) * dx)


def desired_state_v(test, t):
    space_v = test.function_space()
    mesh = space_v.mesh()
    X = SpatialCoordinate(mesh)
    x = X[0] - 1.0
    y = X[1] - 1.0

    a = (100.0 / 49.0) ** 2
    b = (100.0 / 99.0) ** 2

    c_1 = 1.0 - sqrt(a * ((x - 0.5) ** 2) + b * (y ** 2))
    c_2 = 1.0 - sqrt(a * ((x + 0.5) ** 2) + b * (y ** 2))
    v_d = Function(space_v, name="v_d")
    v_d.interpolate(
        ufl.conditional(c_1 >= 0.0,
            c_1 * cos(pi * t / 2.0) * as_vector((b * y,
                                                 -a * (x - 0.5))),
            ufl.conditional(c_2 >= 0.0,
                c_2 * cos(pi * t / 2.0) * as_vector((-b * y,
                                                     a * (x + 0.5))),
                as_vector((0.0, 0.0)))),
    )
    return inner(v_d, test) * dx, v_d

e_min_v = 0.3924
e_max_v = 2.0598
sp_11block = {
    "ksp_type": "chebyshev",
    "pc_type": "jacobi",
    "ksp_chebyshev_eigenvalues": f"{e_min_v:.16e}, {e_max_v:.16e}",
    "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
    "ksp_chebyshev_esteig_steps": 0,
    "ksp_chebyshev_esteig_noisy": False,
    "ksp_max_it": 20,
    "ksp_atol": 0.0,
    "ksp_rtol": 0.0}

e_min_p = 0.5
e_max_p = 2.0
sp_M_p = {
    "ksp_type": "chebyshev",
    "pc_type": "jacobi",
    "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",
    "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
    "ksp_chebyshev_esteig_steps": 0,
    "ksp_chebyshev_esteig_noisy": False,
    "ksp_max_it": 20,
    "ksp_atol": 0.0,
    "ksp_rtol": 0.0}

auxiliary_sp = {
    "sp_11block": sp_11block,
    "sp_M_p": sp_M_p}

nu_range = [1./100.]
beta_range = [1.0e-3, 1.0e-4, 1.0e-5]

for nu in nu_range:
    for beta in beta_range:
        for mesh_size in range(2, 7):
            mesh = RectangleMesh(2**mesh_size, 2**mesh_size, 2.0, 2.0)

            space_v = VectorFunctionSpace(mesh, "Lagrange", 2)
            space_p = FunctionSpace(mesh, "Lagrange", 1)

            control_instationary = Instationary(
                space_v, forw_diff_operator_v, desired_state=desired_state_v,
                time_interval=(0.0, 2.0), n_t=2**mesh_size,
                beta=beta, bcs_v=my_DirichletBC_t_v)

            print(f'Beta: {beta:.16e}')
            print('Mesh-size:', mesh_size)
            t0 = perf_counter()
            control_instationary.incompressible_non_linear_solve(
                ConstantNullspace(), space_p=space_p,
                auxiliary_sp=auxiliary_sp, print_error_non_linear=True,
                create_output=False)
            t1 = perf_counter()
            total_time = t1 - t0
            print(f'Elapsed time: {total_time:.16e}')
            print('\n', flush=True)
