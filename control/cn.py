from firedrake import Cofunction, Function, MixedFunctionSpace

__all__ = \
    [
        "apply_T_1",
        "apply_T_2",
        "apply_T_1_inv",
        "apply_T_2_inv"
    ]


# definition of application of T_1 and T_2 for CN discretization
def apply_T_1(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    full_space_v = MixedFunctionSpace(flattened_space)

    if isinstance(x_old, Function):
        x_new = Function(full_space_v)
    else:
        x_new = Cofunction(full_space_v.dual())
    x_new.assign(x_old)

    for i in range(n_blocks - 1):
        with x_new.sub(i).dat.vec as b_v, \
                x_old.sub(i + 1).dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

    return x_new


def apply_T_2(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    full_space_v = MixedFunctionSpace(flattened_space)

    if isinstance(x_old, Function):
        x_new = Function(full_space_v)
    else:
        x_new = Cofunction(full_space_v.dual())
    x_new.assign(x_old)

    for i in range(1, n_blocks):
        with x_new.sub(i).dat.vec as b_v, \
                x_old.sub(i - 1).dat.vec_ro as b_1_v:
            b_v.axpy(1.0, b_1_v)

    return x_new


# definition of application of T_1^-1 and T_2^-1 for CN discretization
def apply_T_1_inv(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    full_space_v = MixedFunctionSpace(flattened_space)

    if isinstance(x_old, Function):
        x_new = Function(full_space_v)
    else:
        x_new = Cofunction(full_space_v.dual())
    x_new.assign(x_old)

    for i in range(n_blocks - 2, -1, -1):
        with x_new.sub(i).dat.vec as b_v, \
                x_new.sub(i + 1).dat.vec_ro as b_1_v:
            b_v.axpy(-1.0, b_1_v)

    return x_new


def apply_T_2_inv(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    full_space_v = MixedFunctionSpace(flattened_space)

    if isinstance(x_old, Function):
        x_new = Function(full_space_v)
    else:
        x_new = Cofunction(full_space_v.dual())
    x_new.assign(x_old)

    for i in range(1, n_blocks):
        with x_new.sub(i).dat.vec as b_v, \
                x_new.sub(i - 1).dat.vec_ro as b_1_v:
            b_v.axpy(-1.0, b_1_v)

    return x_new
