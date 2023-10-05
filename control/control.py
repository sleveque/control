#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from firedrake.functionspaceimpl import WithGeometry as FunctionSpaceBase

import ufl

from preconditioner.preconditioner import *

import petsc4py.PETSc as PETSc

from collections.abc import Sequence


__all__ = \
    [
        "Control"
    ]


_error_flag = [False]


# definition of application of T_1 and T_2
def apply_T_1(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    mixed_element = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element)

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
    mixed_element = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element)

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


# definition of application of T_1^-1 and T_2^-1
def apply_T_1_inv(x_old, space_v, n_blocks):
    flattened_space = tuple(space_v for i in range(n_blocks))
    mixed_element = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element)

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
    mixed_element = ufl.classes.MixedElement(
        *[space.ufl_element() for space in flattened_space])
    full_space_v = FunctionSpace(space_v.mesh(), mixed_element)

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


class Control:
    class Stationary:
        def __init__(self, space_v,
                     forward_form, desired_state, force_function, *,
                     beta=10.0**-3, space_p=None, Gauss_Newton=False,
                     J_v=None, J_u=None,
                     bcs_v=None):
            if bcs_v is None:
                bcs_v = ()
            elif not isinstance(bcs_v, Sequence):
                bcs_v = (bcs_v,)
            else:
                bcs_v = tuple(bcs_v)

            if not isinstance(space_v, FunctionSpaceBase):
                raise TypeError("Space must be a primal space")
            if space_p is not None \
                    and not isinstance(space_p, FunctionSpaceBase):
                raise TypeError("Space must be a primal space")

            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
            if J_v is None:
                def J_v(v_trial, v_test):
                    return inner(v_trial, v_test) * dx
                J_v = J_v

                M_v = inner(v_trial, v_test) * dx
            else:
                M_v = J_v(v_trial, v_test)
            self._J_v = J_v

            if J_u is None:
                def J_u(v_trial, v_test):
                    return inner(v_trial, v_test) * dx
                J_u = J_u

                M_zeta = inner(v_trial, v_test) * dx
            else:
                M_zeta = J_u(v_trial, v_test)
            self._J_u = J_u

            self._space_v = space_v
            self._space_p = space_p
            self._forward_form = forward_form
            self._desired_state = desired_state
            self._force_function = force_function
            self._beta = beta
            self._bcs_v = bcs_v

            self._M_v = M_v
            self._M_zeta = M_zeta
            self._M_p = None
            self._M_mu = None

            self._Gauss_Newton = Gauss_Newton

            v = Function(space_v, name="v")
            zeta = Function(space_v, name="zeta")

            for bc in self._bcs_v:
                bc.apply(v)

            self._v = v
            self._zeta = zeta

            if space_p is not None:
                p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

                M_p = J_v(p_trial, p_test)
                M_mu = J_u(p_trial, p_test)

                self._M_p = M_p
                self._M_mu = M_mu

                p = Function(space_p, name="p")
                mu = Function(space_p, name="mu")

                self._p = p
                self._mu = mu

        def set_space_v(self, space_v, *, v=None, zeta=None,
                        bcs_v_new=False, bcs_v=None):
            self._space_v = space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            M_v = self._J_v(v_trial, v_test)
            M_zeta = self._J_u(v_trial, v_test)

            self._M_v = M_v
            self._M_zeta = M_zeta

            if v is None:
                v = Function(space_v, name="v")
            else:
                if v.function_space() != space_v:
                    raise ValueError("Unexpected space")

            if bcs_v_new:
                if bcs_v is None:
                    bcs_v = ()
                elif not isinstance(bcs_v, Sequence):
                    bcs_v = (bcs_v,)
                else:
                    bcs_v = tuple(bcs_v)
                self._bcs_v = bcs_v

            for bc in self._bcs_v:
                bc.apply(v)

            if zeta is None:
                zeta = Function(space_v, name="zeta")
            else:
                if zeta.function_space() != space_v:
                    raise ValueError("Unexpected space")
                else:
                    bcs_zeta = homogenize(self._bcs_v)
                    for bc in bcs_zeta:
                        bc.apply(zeta)

            self._v = v
            self._zeta = zeta

        def set_space_p(self, space_p, *, p=None, mu=None):
            self._space_p = space_p
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            M_p = self._J_v(p_trial, p_test)
            M_mu = self._J_u(p_trial, p_test)

            self._M_p = M_p
            self._M_mu = M_mu

            if p is None:
                p = Function(space_p, name="p")
            else:
                if p.function_space() != space_p:
                    raise ValueError("Unexpected space")

            if mu is None:
                mu = Function(space_p, name="mu")
            else:
                if mu.function_space() != space_p:
                    raise ValueError("Unexpected space")

            self._p = p
            self._mu = mu

        def set_forward_form(self, forward_form):
            self._forward_form = forward_form

        def set_desired_state(self, desired_state):
            self._desired_state = desired_state

        def set_force_function(self, force_function):
            self._force_function = force_function

        def set_beta(self, beta):
            self._beta = beta

        def set_J_v(self, J_v):
            self._J_v = J_v
            v_test = TestFunction(self._space_v)
            v_trial = TrialFunction(self._space_v)
            self._M_v = J_v(v_trial, v_test)

            if self._space_p is not None:
                p_test = TestFunction(self._space_p)
                p_trial = TrialFunction(self._space_p)
                self._M_p = J_v(p_trial, p_test)

        def set_J_u(self, J_u):
            self._J_u = J_u
            v_test = TestFunction(self._space_v)
            v_trial = TrialFunction(self._space_v)
            self._M_zeta = J_u(v_trial, v_test)

            if self._space_p is not None:
                p_test = TestFunction(self._space_p)
                p_trial = TrialFunction(self._space_p)
                self._M_mu = J_u(p_trial, p_test)

        def set_bcs_v(self, bcs_v, space_v=None):
            if space_v is None:
                if bcs_v is None:
                    bcs_v = ()
                elif not isinstance(bcs_v, Sequence):
                    bcs_v = (bcs_v,)
                else:
                    bcs_v = tuple(bcs_v)

                self._bcs_v = bcs_v
            else:
                self.set_space_v(space_v, bcs_v_new=True, bcs_v=bcs_v)

        def set_Gauss_Newton(self, Gauss_Newton=True):
            self._Gauss_Newton = Gauss_Newton

        def set_v(self, v_new):
            if v_new.function_space() != self._space_v:
                raise ValueError("Unexpected space")
            else:
                v_help = Function(v_new.function_space(), name="v")
                v_help.assign(v_new)
                for bc in self._bcs_v:
                    bc.apply(v_help)
                self._v.assign(v_help)

        def set_zeta(self, zeta_new):
            if zeta_new.function_space() != self._space_v:
                raise ValueError("Unexpected space")
            else:
                zeta_help = Function(zeta_new.function_space(), name="zeta")
                zeta_help.assign(zeta_new)
                bcs_zeta = homogenize(self._bcs_v)
                for bc in bcs_zeta:
                    bc.apply(zeta_help)
                self._zeta.assign(zeta_help)

        def set_p(self, p_new):
            if self._space_p is not None:
                if p_new.function_space() != self._space_p:
                    raise ValueError("Unexpected space")
                else:
                    self._p.assign(p_new)
            else:
                raise ValueError("Undefined space_p: unable to assign value")

        def set_mu(self, mu_new):
            if self._space_p is not None:
                if mu_new.function_space() != self._space_p:
                    raise ValueError("Unexpected space")
                else:
                    self._mu.assign(mu_new)
            else:
                raise ValueError("Undefined space_p: unable to assign value")

        def print_error(self, v_test):
            v_d, true_v = self._desired_state(v_test)
            v_err = self._v - true_v
            print('estimated error in the L2-norm: ',
                  sqrt(abs(assemble(inner(v_err, v_err) * dx))))

        def construct_D_v(self, v_trial, v_test, v_old):
            if not self._Gauss_Newton:
                D_v = self._forward_form(v_trial, v_test, v_old)
            else:
                D_v = ufl.derivative(self._forward_form(v_trial,
                                                        v_test,
                                                        v_old),
                                     v_old,
                                     v_trial)

            return D_v

        def construct_f(self, inhomogeneous_bcs_v, v_test,
                        D_v, v_inhom, bcs_v):
            if inhomogeneous_bcs_v:
                f = assemble(self._force_function(v_test)
                             - action(D_v, v_inhom))
                for bc in bcs_v:
                    bc.apply(f)
            else:
                f = assemble(self._force_function(v_test))

            return f

        def construct_v_d(self, v_test, inhomogeneous_bcs_v, v_inhom, bcs_v):
            v_d, true_v = self._desired_state(v_test)
            if inhomogeneous_bcs_v:
                v_d = assemble(v_d - action(self._M_v, v_inhom))
                for bc in bcs_v:
                    bc.apply(v_d)
            else:
                v_d = assemble(v_d)

            return v_d, true_v

        def construct_pc(self, Multigrid, lambda_v_bounds,
                         bcs_v, bcs_zeta, D_v, D_zeta):
            beta = self._beta

            # definition of preconditioners
            def pc_linear(u_0, u_1, b_0, b_1):
                if Multigrid:
                    solver_0 = LinearSolver(
                        assemble(self._M_v,
                                 bcs=bcs_v),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                else:
                    if lambda_v_bounds is not None:
                        e_min = lambda_v_bounds[0]
                        e_max = lambda_v_bounds[1]
                        solver_0 = LinearSolver(
                            assemble(self._M_v,
                                     bcs=bcs_v),
                            solver_parameters={
                                "ksp_type": "chebyshev",
                                "pc_type": "jacobi",
                                "ksp_chebyshev_eigenvalues": f"{e_min:.16e}, {e_max:.16e}",  # noqa: E501
                                "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                                "ksp_chebyshev_esteig_steps": 0,
                                "ksp_chebyshev_esteig_noisy": False,
                                "ksp_max_it": 20,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                    else:
                        solver_0 = LinearSolver(
                            assemble(self._M_v,
                                     bcs=bcs_v),
                            solver_parameters={"ksp_type": "preonly",
                                               "pc_type": "jacobi",
                                               "ksp_max_it": 1,
                                               "ksp_atol": 0.0,
                                               "ksp_rtol": 0.0})

                solver_1 = LinearSolver(
                    assemble(D_v + (1.0 / beta**0.5) * self._M_v,
                             bcs=bcs_zeta),
                    solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "hypre",
                                       "pc_hypre_type": "boomeramg",
                                       "ksp_max_it": 1,
                                       "pc_hypre_boomeramg_max_iter": 2,
                                       "ksp_atol": 0.0,
                                       "ksp_rtol": 0.0})

                solver_2 = LinearSolver(
                    assemble(D_zeta + (1.0 / beta**0.5) * self._M_zeta,
                             bcs=bcs_zeta),
                    solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "hypre",
                                       "pc_hypre_type": "boomeramg",
                                       "ksp_max_it": 1,
                                       "pc_hypre_boomeramg_max_iter": 2,
                                       "ksp_atol": 0.0,
                                       "ksp_rtol": 0.0})

                # solving for the (1,1)-block
                try:
                    solver_0.solve(u_0, b_0.copy(deepcopy=True))
                except ConvergenceError:
                    assert solver_0.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                # u_1 = - b_1 + D_v * u_0
                b = assemble(action(D_v, u_0))
                with b.dat.vec as b_v, \
                        b_1.dat.vec_ro as b_1_v:
                    b_v.axpy(-1.0, b_1_v)

                # solving for the Schur complement approximation
                # first solve
                for bc in bcs_zeta:
                    bc.apply(b)
                try:
                    solver_1.solve(u_1, b.copy(deepcopy=True))
                except ConvergenceError:
                    assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                del b

                # second solve
                b = assemble(action(self._M_v, u_1))
                for bc in bcs_zeta:
                    bc.apply(b)
                try:
                    solver_2.solve(u_1, b.copy(deepcopy=True))
                except ConvergenceError:
                    assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                del b

            return pc_linear

        def non_linear_res_eval(self, space_v, v_d, f, v_old, zeta_old,
                                D_v, D_zeta, M_zeta, bcs_v, bcs_zeta):
            rhs_0 = Cofunction(space_v.dual(), name="rhs_0")
            rhs_1 = Cofunction(space_v.dual(), name="rhs_1")

            rhs_0.assign(v_d)
            rhs_1.assign(f)

            b = assemble(action(self._M_v, v_old))
            with b.dat.vec_ro as b_v, \
                    rhs_0.dat.vec as b_1_v:
                b_1_v.axpy(-1.0, b_v)
            del b
            b = assemble(action(D_zeta, zeta_old))
            with b.dat.vec_ro as b_v, \
                    rhs_0.dat.vec as b_1_v:
                b_1_v.axpy(-1.0, b_v)
            del b

            b = assemble(action(D_v, v_old))
            with b.dat.vec_ro as b_v, \
                    rhs_1.dat.vec as b_1_v:
                b_1_v.axpy(-1.0, b_v)
            del b
            b = assemble(action(M_zeta, zeta_old))
            with b.dat.vec_ro as b_v, \
                    rhs_1.dat.vec as b_1_v:
                b_1_v.axpy(-1.0, b_v)
            del b

            for bc in bcs_v:
                bc.apply(rhs_0)
            for bc in bcs_zeta:
                bc.apply(rhs_1)

            return rhs_0, rhs_1

        def linear_solve(self, *,
                         P=None, solver_parameters=None, Multigrid=False,
                         lambda_v_bounds=None, v_d=None, f=None,
                         print_error=True, create_output=True,
                         plots=True):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            inhomogeneous_bcs_v = False
            for bc in self._bcs_v:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v)
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v
            bcs_zeta = bcs_v

            nullspace_v = DirichletBCNullspace(bcs_v)
            nullspace_zeta = DirichletBCNullspace(bcs_zeta)

            beta = self._beta

            v_old = Function(space_v, name="v_old")
            v_old.assign(self._v)

            D_v = self.construct_D_v(v_trial, v_test, v_old)
            D_zeta = adjoint(D_v)

            if inhomogeneous_bcs_v:
                v_inhom = Function(space_v)
                for bc in bcs_v_help:
                    bc.apply(v_inhom)
            else:
                v_inhom = None

            if f is None:
                f = self.construct_f(inhomogeneous_bcs_v, v_test,
                                     D_v, v_inhom, bcs_v)

            if v_d is None:
                v_d, true_v = self.construct_v_d(v_test, inhomogeneous_bcs_v,
                                                 v_inhom, bcs_v)

            if P is None:
                pc_fn = self.construct_pc(Multigrid, lambda_v_bounds,
                                          bcs_v, bcs_zeta, D_v, D_zeta)
            else:
                pc_fn = P

            block_00 = {}
            block_00[(0, 0)] = self._M_v
            block_01 = {}
            block_01[(0, 0)] = D_zeta
            block_10 = {}
            block_10[(0, 0)] = D_v
            block_11 = {}
            block_11[(0, 0)] = -(1.0 / beta) * self._M_zeta

            system = MultiBlockSystem(
                space_v, space_v,
                block_00=block_00, block_01=block_01,
                block_10=block_10, block_11=block_11,
                nullspace_0=(nullspace_v,), nullspace_1=(nullspace_zeta,))

            if solver_parameters is None:
                solver_parameters = {"linear_solver": "gmres",
                                     "gmres_restart": 10,
                                     "maximum_iterations": 50,
                                     "relative_tolerance": 1.0e-6,
                                     "absolute_tolerance": 0.0,
                                     "monitor_convergence": print_error}

            v = Function(space_v, name="v")
            zeta = Function(space_v, name="zeta")

            system.solve(
                v, zeta, v_d, f,
                solver_parameters=solver_parameters,
                pc_fn=pc_fn)

            if inhomogeneous_bcs_v:
                with v.dat.vec as b_v, \
                        v_inhom.dat.vec_ro as b_1_v:
                    b_v.axpy(1.0, b_1_v)

            self.set_v(v)
            self.set_zeta(zeta)

            if print_error:
                self.print_error(v_test)

            if create_output:
                v_output = File("v.pvd")
                v_output.write(v)

                zeta_output = File("zeta.pvd")
                zeta_output.write(zeta)

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(v)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(zeta)

            if plots:
                try:
                    import matplotlib.pyplot as plt
                    fig_v, axes = plt.subplots()
                    colors = tripcolor(v, axes=axes)
                    fig_v.colorbar(colors)
                    fig_zeta, axes = plt.subplots()
                    colors = tripcolor(zeta, axes=axes)
                    fig_zeta.colorbar(colors)
                    fig_true_v, axes = plt.subplots()
                    colors = tripcolor(true_v, axes=axes)
                    fig_true_v.colorbar(colors)
                    plt.show()
                except Exception as e:
                    warning("Cannot plot figure. Error msg: '%s'" % e)

        def non_linear_solve(self, *,
                             P=None, solver_parameters=None,
                             Multigrid=False,
                             lambda_v_bounds=None,
                             max_non_linear_iter=10,
                             relative_non_linear_tol=10.0**-5,
                             absolute_non_linear_tol=10.0**-8,
                             print_error_linear=False,
                             print_error_non_linear=True,
                             create_output=True,
                             plots=True):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            inhomogeneous_bcs_v = False
            for bc in self._bcs_v:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v)
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v
            bcs_zeta = bcs_v

            beta = self._beta

            v_old = Function(space_v, name="v_old")
            zeta_old = Function(space_v, name="zeta_old")
            delta_v = Function(space_v, name="delta_v")
            delta_zeta = Function(space_v, name="delta_zeta")

            v_old.assign(self._v)
            zeta_old.assign(self._zeta)

            D_v = self.construct_D_v(v_trial, v_test, v_old)
            D_zeta = adjoint(D_v)
            M_zeta = -(1.0 / beta) * self._M_zeta

            f = assemble(self._force_function(v_test))

            v_d, true_v = self._desired_state(v_test)
            v_d = assemble(v_d)

            rhs_0, rhs_1 = self.non_linear_res_eval(
                space_v, v_d, f, v_old, zeta_old,
                D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

            rhs = Cofunction((space_v * space_v).dual(), name="rhs")
            rhs.sub(0).assign(rhs_0)
            rhs.sub(1).assign(rhs_1)

            with rhs.dat.vec_ro as b_v:
                norm_0 = b_v.norm()
            norm_k = norm_0

            k = 0

            while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):  # noqa: E501
                self.linear_solve(P=P, solver_parameters=solver_parameters,
                                  Multigrid=Multigrid,
                                  lambda_v_bounds=lambda_v_bounds,
                                  v_d=rhs_0, f=rhs_1,
                                  print_error=print_error_linear,
                                  create_output=False,
                                  plots=False)

                delta_v.assign(self._v)
                delta_zeta.assign(self._zeta)

                with delta_v.dat.vec_ro as b_v, \
                        v_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                if inhomogeneous_bcs_v:
                    for bc in bcs_v_help:
                        bc.apply(v_old)
                self.set_v(v_old)

                with delta_zeta.dat.vec_ro as b_v, \
                        zeta_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                for bc in bcs_zeta:
                    bc.apply(zeta_old)
                self.set_zeta(zeta_old)

                D_v = self.construct_D_v(v_trial, v_test, v_old)
                D_zeta = adjoint(D_v)

                rhs_0, rhs_1 = self.non_linear_res_eval(
                    space_v, v_d, f, v_old, zeta_old,
                    D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

                rhs.sub(0).assign(rhs_0)
                rhs.sub(1).assign(rhs_1)

                with rhs.dat.vec_ro as b_v:
                    norm_k = b_v.norm()

                k += 1
                if k + 1 > max_non_linear_iter:
                    break

            if print_error_non_linear:
                if norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol:  # noqa: E501
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                    print('number of non-linear iterations: ', k)
                else:
                    print('the non-linear iteration did not converge')
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                self.print_error(v_test)

            if create_output:
                v_output = File("v.pvd")
                v_output.write(self._v)

                zeta_output = File("zeta.pvd")
                zeta_output.write(self._zeta)

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(self._v)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(self._zeta)

            if plots:
                try:
                    import matplotlib.pyplot as plt
                    fig_v, axes = plt.subplots()
                    colors = tripcolor(self._v, axes=axes)
                    fig_v.colorbar(colors)
                    fig_zeta, axes = plt.subplots()
                    colors = tripcolor(self._zeta, axes=axes)
                    fig_zeta.colorbar(colors)
                    fig_true_v, axes = plt.subplots()
                    colors = tripcolor(true_v, axes=axes)
                    fig_true_v.colorbar(colors)
                    plt.show()
                except Exception as e:
                    warning("Cannot plot figure. Error msg: '%s'" % e)

        def incompressible_linear_solve(self, nullspace_p, *, space_p=None,
                                        P=None, solver_parameters=None,
                                        Multigrid=False,
                                        lambda_v_bounds=None,
                                        lambda_p_bounds=None,
                                        v_d=None, f=None,
                                        div_v=None, div_zeta=None,
                                        print_error=True, create_output=True,
                                        plots=True):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
            if space_p is None:
                if self._space_p is not None:
                    space_p = self._space_p
                else:
                    raise ValueError("Undefined space_p")
            else:
                self.set_space_p(space_p)
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            inhomogeneous_bcs_v = False
            for bc in self._bcs_v:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v)
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v
            bcs_zeta = bcs_v

            nullspace_v = DirichletBCNullspace(bcs_v)
            nullspace_zeta = DirichletBCNullspace(bcs_zeta)

            space_0 = FunctionSpace(
                space_v.mesh(), space_v.ufl_element() * space_v.ufl_element())
            space_1 = FunctionSpace(
                space_p.mesh(), space_p.ufl_element() * space_p.ufl_element())

            beta = self._beta

            v_old = Function(space_v, name="v_old")
            v_old.assign(self._v)

            M_zeta = -(1.0 / beta) * self._M_zeta
            D_v = self.construct_D_v(v_trial, v_test, v_old)
            D_zeta = adjoint(D_v)

            B = - inner(div(v_trial), p_test) * dx
            B_T = - inner(p_trial, div(v_test)) * dx

            if inhomogeneous_bcs_v:
                v_inhom = Function(space_v)
                for bc in bcs_v_help:
                    bc.apply(v_inhom)
            else:
                v_inhom = None

            if f is None:
                f = self.construct_f(inhomogeneous_bcs_v, v_test,
                                     D_v, v_inhom, bcs_v)

            if v_d is None:
                v_d, true_v = self.construct_v_d(v_test, inhomogeneous_bcs_v,
                                                 v_inhom, bcs_v)

            if div_v is None:
                div_v = Function(space_p)
                if inhomogeneous_bcs_v:
                    div_v = assemble(- action(B, v_inhom))

            if div_zeta is None:
                div_zeta = Cofunction(space_p.dual())

            b_0 = Cofunction(space_0.dual(), name="b_0")
            b_1 = Cofunction(space_1.dual(), name="b_1")

            b_0.sub(0).assign(v_d)
            b_0.sub(1).assign(f)
            b_1.sub(0).assign(div_v)
            b_1.sub(1).assign(div_zeta)

            block_00 = {}
            block_00[(0, 0)] = self._M_v
            block_00[(0, 1)] = D_zeta
            block_00[(1, 0)] = D_v
            block_00[(1, 1)] = M_zeta
            block_01 = {}
            block_01[(0, 0)] = B_T
            block_01[(0, 1)] = None
            block_01[(1, 0)] = None
            block_01[(1, 1)] = B_T
            block_10 = {}
            block_10[(0, 0)] = B
            block_10[(0, 1)] = None
            block_10[(1, 0)] = None
            block_10[(1, 1)] = B
            block_11 = {}
            block_11[(0, 0)] = None
            block_11[(0, 1)] = None
            block_11[(1, 0)] = None
            block_11[(1, 1)] = None

            nullspace_0 = (nullspace_v, nullspace_zeta)
            nullspace_1 = (nullspace_p, nullspace_p)

            system = MultiBlockSystem(
                space_v, space_p,
                block_00=block_00, block_01=block_01,
                block_10=block_10, block_11=block_11,
                n_blocks_00=2, n_blocks_11=2,
                nullspace_0=nullspace_0, nullspace_1=nullspace_1)

            if P is None:
                block_00_int = {}
                block_00_int[(0, 0)] = self._M_v
                block_01_int = {}
                block_01_int[(0, 0)] = D_zeta
                block_10_int = {}
                block_10_int[(0, 0)] = D_v
                block_11_int = {}
                block_11_int[(0, 0)] = M_zeta

                K_p = inner(grad(p_trial), grad(p_test)) * dx
                M_p = inner(p_trial, p_test) * dx

                solver_K_p = LinearSolver(
                    assemble(K_p),
                    solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "hypre",
                                       "pc_hypre_type": "boomeramg",
                                       "ksp_max_it": 1,
                                       "pc_hypre_boomeramg_max_iter": 1,
                                       "ksp_atol": 0.0,
                                       "ksp_rtol": 0.0})

                if lambda_p_bounds is not None:
                    e_min_p = lambda_p_bounds[0]
                    e_max_p = lambda_p_bounds[1]
                    solver_M_p = LinearSolver(
                        assemble(M_p),
                        solver_parameters={
                            "ksp_type": "chebyshev",
                            "pc_type": "jacobi",
                            "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",  # noqa: E501
                            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                            "ksp_chebyshev_esteig_steps": 0,
                            "ksp_chebyshev_esteig_noisy": False,
                            "ksp_max_it": 20,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                else:
                    solver_M_p = LinearSolver(
                        assemble(M_p),
                        solver_parameters={"ksp_type": "preonly",
                                           "pc_type": "jacobi",
                                           "ksp_max_it": 1,
                                           "ksp_atol": 0.0,
                                           "ksp_rtol": 0.0})

                if self._M_p is not None:
                    block_00_p = self._M_p
                else:
                    block_00_p = self._J_v(p_trial, p_test)
                block_10_p = self.construct_D_v(p_trial, p_test, v_old)
                block_01_p = adjoint(block_10_p)
                if self._M_mu is not None:
                    block_11_p = - (1.0 / beta) * self._M_mu
                else:
                    block_11_p = - (1.0 / beta) * self._J_u(p_trial, p_test)  # noqa: E501

                def pc_fn(u_0, u_1, b_0, b_1):
                    b_0_help = Cofunction(space_v.dual())
                    b_1_help = Cofunction(space_v.dual())

                    b_0_help.assign(b_0.sub(0))
                    b_1_help.assign(b_0.sub(1))

                    inner_system = MultiBlockSystem(
                        space_v, space_v,
                        block_00=block_00_int, block_01=block_01_int,
                        block_10=block_10_int, block_11=block_11_int,
                        nullspace_0=(nullspace_v,),
                        nullspace_1=(nullspace_zeta,))

                    inner_solver_parameters = {"preconditioner": True,
                                               "linear_solver": "gmres",
                                               "maximum_iterations": 5,
                                               "relative_tolerance": 0.0,
                                               "absolute_tolerance": 0.0,
                                               "monitor_convergence": False}

                    v_help = Function(space_v)
                    zeta_help = Function(space_v)

                    inner_pc_fn = self.construct_pc(Multigrid, lambda_v_bounds,
                                                    bcs_v, bcs_zeta,
                                                    D_v, D_zeta)

                    try:
                        inner_ksp_solver = inner_system.solve(
                            v_help, zeta_help, b_0_help, b_1_help,
                            solver_parameters=inner_solver_parameters,
                            pc_fn=inner_pc_fn)
                    except ConvergenceError:
                        assert inner_ksp_solver.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                    del b_0_help
                    del b_1_help

                    u_0.sub(0).assign(v_help)
                    u_0.sub(1).assign(zeta_help)

                    # u_1 = - b_1 + block_10 * u_0
                    b_0_help = Cofunction(space_p.dual())
                    b_1_help = Cofunction(space_p.dual())
                    b_0_help.assign(assemble(action(B, v_help)))
                    b_1_help.assign(assemble(action(B, zeta_help)))
                    with b_0_help.dat.vec as b_v, \
                            b_1.sub(0).dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    with b_1_help.dat.vec as b_v, \
                            b_1.sub(1).dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    del v_help
                    del zeta_help

                    # solving for the Schur complement approximation
                    try:
                        solver_K_p.solve(u_1.sub(0),
                                         b_0_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    try:
                        solver_K_p.solve(u_1.sub(1),
                                         b_1_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                    del b_0_help
                    del b_1_help

                    b_0_help = Cofunction(space_p.dual())
                    b_1_help = Cofunction(space_p.dual())
                    b_c_0_help = Function(space_p)
                    b_c_1_help = Function(space_p)
                    b_c_0_help.assign(u_1.sub(0))
                    b_c_1_help.assign(u_1.sub(1))
                    b_0_help.assign(assemble(
                        action(block_00_p, b_c_0_help)
                        + action(block_01_p, b_c_1_help)))
                    b_1_help.assign(assemble(
                        action(block_10_p, b_c_0_help)
                        + action(block_11_p, b_c_1_help)))
                    del b_c_0_help
                    del b_c_1_help

                    try:
                        solver_M_p.solve(u_1.sub(0),
                                         b_0_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    try:
                        solver_M_p.solve(u_1.sub(1),
                                         b_1_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                    del b_0_help
                    del b_1_help
            else:
                pc_fn = P

            if solver_parameters is None:
                solver_parameters = {"linear_solver": "fgmres",
                                     "fgmres_restart": 10,
                                     "maximum_iterations": 50,
                                     "relative_tolerance": 1.0e-6,
                                     "absolute_tolerance": 0.0,
                                     "monitor_convergence": print_error}

            u_0_sol = Function(space_0)
            u_1_sol = Function(space_1)

            system.solve(
                u_0_sol, u_1_sol, b_0, b_1,
                solver_parameters=solver_parameters,
                pc_fn=pc_fn)

            v = Function(space_v, name="v")
            zeta = Function(space_v, name="zeta")
            p = Function(space_p, name="p")
            mu = Function(space_p, name="mu")

            v.assign(u_0_sol.sub(0))
            zeta.assign(u_0_sol.sub(1))

            if inhomogeneous_bcs_v:
                with v.dat.vec as b_v, \
                        v_inhom.dat.vec_ro as b_1_v:
                    b_v.axpy(1.0, b_1_v)

            p.assign(u_1_sol.sub(1))
            mu.assign(u_1_sol.sub(0))

            self.set_v(v)
            self.set_zeta(zeta)

            self.set_p(p)
            self.set_mu(mu)

            if print_error:
                self.print_error(v_test)

            if create_output:
                v_output = File("v.pvd")
                v_output.write(v)

                zeta_output = File("zeta.pvd")
                zeta_output.write(zeta)

                p_output = File("p.pvd")
                p_output.write(p)

                mu_output = File("mu.pvd")
                mu_output.write(mu)

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(v)

                with CheckpointFile("p.h5", "w") as h:
                    h.save_function(p)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(zeta)

                with CheckpointFile("mu.h5", "w") as h:
                    h.save_function(mu)

            if plots:
                try:
                    import matplotlib.pyplot as plt
                    fig_v, axes = plt.subplots()
                    colors = tripcolor(v, axes=axes)
                    fig_v.colorbar(colors)
                    fig_p, axes = plt.subplots()
                    colors = tripcolor(p, axes=axes)
                    fig_p.colorbar(colors)
                    fig_zeta, axes = plt.subplots()
                    colors = tripcolor(zeta, axes=axes)
                    fig_zeta.colorbar(colors)
                    fig_mu, axes = plt.subplots()
                    colors = tripcolor(mu, axes=axes)
                    fig_mu.colorbar(colors)
                    fig_true_v, axes = plt.subplots()
                    colors = tripcolor(true_v, axes=axes)
                    fig_true_v.colorbar(colors)
                    plt.show()
                except Exception as e:
                    warning("Cannot plot figure. Error msg: '%s'" % e)

        def incompressible_non_linear_solve(self, nullspace_p, *, space_p=None,
                                            P=None, solver_parameters=None,
                                            Multigrid=False,
                                            lambda_v_bounds=None,
                                            lambda_p_bounds=None,
                                            max_non_linear_iter=10,
                                            relative_non_linear_tol=10.0**-5,
                                            absolute_non_linear_tol=10.0**-8,
                                            print_error_linear=False,
                                            print_error_non_linear=True,
                                            create_output=True,
                                            plots=True):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
            if space_p is None:
                if self._space_p is not None:
                    space_p = self._space_p
                else:
                    raise ValueError("Undefined space_p")
            else:
                self.set_space_p(space_p)
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            space_0 = FunctionSpace(
                space_v.mesh(), space_v.ufl_element() * space_v.ufl_element())
            space_1 = FunctionSpace(
                space_p.mesh(), space_p.ufl_element() * space_p.ufl_element())

            inhomogeneous_bcs_v = False
            for bc in self._bcs_v:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v)
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v
            bcs_zeta = bcs_v

            beta = self._beta

            v_old = Function(space_v, name="v_old")
            zeta_old = Function(space_v, name="zeta_old")
            delta_v = Function(space_v, name="delta_v")
            delta_zeta = Function(space_v, name="delta_zeta")

            p_old = Function(space_p, name="p_old")
            mu_old = Function(space_p, name="mu_old")
            delta_p = Function(space_p, name="delta_p")
            delta_mu = Function(space_p, name="delta_mu")

            v_old.assign(self._v)
            zeta_old.assign(self._zeta)
            p_old.assign(self._p)
            mu_old.assign(self._mu)

            D_v = self.construct_D_v(v_trial, v_test, v_old)
            D_zeta = adjoint(D_v)
            M_zeta = -(1.0 / beta) * self._M_zeta

            B = - inner(div(v_trial), p_test) * dx
            B_T = - inner(p_trial, div(v_test)) * dx

            f = assemble(self._force_function(v_test))

            v_d, true_v = self._desired_state(v_test)
            v_d = assemble(v_d)

            def non_linear_res_eval():
                rhs_00 = Cofunction(space_v.dual(), name="rhs_00")
                rhs_01 = Cofunction(space_v.dual(), name="rhs_01")

                rhs_10 = Cofunction(space_p.dual(), name="rhs_10")
                rhs_11 = Cofunction(space_p.dual(), name="rhs_11")

                rhs_0, rhs_1 = self.non_linear_res_eval(
                    space_v, v_d, f, v_old, zeta_old,
                    D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

                rhs_00.assign(rhs_0)
                rhs_01.assign(rhs_1)

                del rhs_0
                del rhs_1

                b = assemble(action(B_T, mu_old))
                with b.dat.vec_ro as b_v, \
                        rhs_00.dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b

                b = assemble(action(B_T, p_old))
                with b.dat.vec_ro as b_v, \
                        rhs_01.dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b

                b = assemble(action(B, v_old))
                with b.dat.vec_ro as b_v, \
                        rhs_10.dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b

                b = assemble(action(B, zeta_old))
                with b.dat.vec_ro as b_v, \
                        rhs_11.dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b

                for bc in bcs_v:
                    bc.apply(rhs_00)
                for bc in bcs_zeta:
                    bc.apply(rhs_01)

                return rhs_00, rhs_01, rhs_10, rhs_11

            rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

            rhs = Cofunction((space_0 * space_1).dual(), name="rhs")
            rhs.sub(0).assign(rhs_00)
            rhs.sub(1).assign(rhs_01)
            rhs.sub(2).assign(rhs_10)
            rhs.sub(3).assign(rhs_11)

            with rhs.dat.vec_ro as b_v:
                norm_0 = b_v.norm()
            norm_k = norm_0

            k = 0

            while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):  # noqa: E501
                self.incompressible_linear_solve(
                    nullspace_p, space_p=space_p, P=P,
                    solver_parameters=solver_parameters,
                    Multigrid=Multigrid,
                    lambda_v_bounds=lambda_v_bounds,
                    lambda_p_bounds=lambda_p_bounds,
                    v_d=rhs_00, f=rhs_01,
                    div_v=rhs_10, div_zeta=rhs_11,
                    print_error=print_error_linear,
                    create_output=False, plots=False)

                delta_v.assign(self._v)
                delta_zeta.assign(self._zeta)
                delta_p.assign(self._p)
                delta_mu.assign(self._mu)

                with delta_v.dat.vec_ro as b_v, \
                        v_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                if inhomogeneous_bcs_v:
                    for bc in bcs_v_help:
                        bc.apply(v_old)
                self.set_v(v_old)

                with delta_zeta.dat.vec_ro as b_v, \
                        zeta_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                self.set_zeta(zeta_old)

                with delta_p.dat.vec_ro as b_v, \
                        p_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                self.set_p(p_old)

                with delta_mu.dat.vec_ro as b_v, \
                        mu_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                self.set_mu(mu_old)

                D_v = self.construct_D_v(v_trial, v_test, v_old)
                D_zeta = adjoint(D_v)

                rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

                rhs.sub(0).assign(rhs_00)
                rhs.sub(1).assign(rhs_01)
                rhs.sub(2).assign(rhs_10)
                rhs.sub(3).assign(rhs_11)

                with rhs.dat.vec_ro as b_v:
                    norm_k = b_v.norm()

                k += 1
                if k + 1 > max_non_linear_iter:
                    break

            if print_error_non_linear:
                if norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol:  # noqa: E501
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                    print('number of non-linear iterations: ', k)
                else:
                    print('the non-linear iteration did not converge')
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                self.print_error(v_test)

            if create_output:
                v_output = File("v.pvd")
                v_output.write(self._v)

                zeta_output = File("zeta.pvd")
                zeta_output.write(self._zeta)

                p_output = File("p.pvd")
                p_output.write(self._p)

                mu_output = File("mu.pvd")
                mu_output.write(self._mu)

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(self._v)

                with CheckpointFile("p.h5", "w") as h:
                    h.save_function(self._p)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(self._zeta)

                with CheckpointFile("mu.h5", "w") as h:
                    h.save_function(self._mu)

            if plots:
                try:
                    import matplotlib.pyplot as plt
                    fig_v, axes = plt.subplots()
                    colors = tripcolor(self._v, axes=axes)
                    fig_v.colorbar(colors)
                    fig_p, axes = plt.subplots()
                    colors = tripcolor(self._p, axes=axes)
                    fig_p.colorbar(colors)
                    fig_zeta, axes = plt.subplots()
                    colors = tripcolor(self._zeta, axes=axes)
                    fig_zeta.colorbar(colors)
                    fig_mu, axes = plt.subplots()
                    colors = tripcolor(self._mu, axes=axes)
                    fig_mu.colorbar(colors)
                    fig_true_v, axes = plt.subplots()
                    colors = tripcolor(true_v, axes=axes)
                    fig_true_v.colorbar(colors)
                    plt.show()
                except Exception as e:
                    warning("Cannot plot figure. Error msg: '%s'" % e)

    class Instationary:
        def __init__(self, space_v,
                     forward_form, desired_state, force_function, *,
                     beta=10.0**-3, space_p=None, Gauss_Newton=False,
                     CN=True, n_t=20,
                     initial_condition=None, time_interval=None,
                     J_v=None, J_u=None,
                     bcs_v=None):
            if not isinstance(space_v, FunctionSpaceBase):
                raise TypeError("Space must be a primal space")
            if space_p is not None \
                    and not isinstance(space_p, FunctionSpaceBase):
                raise TypeError("Space must be a primal space")

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
            if J_v is None:
                def J_v(v_trial, v_test):
                    return inner(v_trial, v_test) * dx
                J_v = J_v

                M_v = inner(v_trial, v_test) * dx
            else:
                M_v = J_v(v_trial, v_test)
            self._J_v = J_v

            if J_u is None:
                def J_u(v_trial, v_test):
                    return inner(v_trial, v_test) * dx
                J_u = J_u

                M_zeta = inner(v_trial, v_test) * dx
            else:
                M_zeta = J_u(v_trial, v_test)
            self._J_u = J_u

            self._space_v = space_v
            self._space_p = space_p
            self._forward_form = forward_form
            self._desired_state = desired_state
            self._force_function = force_function
            self._beta = beta
            self._initial_condition = initial_condition
            if time_interval is not None:
                self._time_interval = time_interval
            else:
                self._time_interval = (0.0, 1.0)
            self._CN = CN
            self._n_t = n_t

            self._f_bcs_v = bcs_v
            full_bcs_v = {}
            if bcs_v is None:
                for i in range(n_t):
                    full_bcs_v[(i)] = ()
            else:
                t_0 = self._time_interval[0]
                T_f = self._time_interval[1]
                tau = (T_f - t_0) / (n_t - 1.0)
                for i in range(n_t):
                    t = t_0 + i * tau
                    bcs_v_i = bcs_v(space_v, t)
                    if not isinstance(bcs_v_i, Sequence):
                        full_bcs_v[(i)] = (bcs_v_i, )
                    else:
                        full_bcs_v[(i)] = tuple(bcs_v_i)
            self._bcs_v = full_bcs_v

            self._M_v = M_v
            self._M_zeta = M_zeta
            self._M_p = None
            self._M_mu = None

            self._Gauss_Newton = Gauss_Newton

            v = Function(full_space_v, name="v")
            zeta = Function(full_space_v, name="zeta")

            if initial_condition is not None:
                v_test = TestFunction(space_v)
                v.sub(0).assign(initial_condition(v_test))

            for i in range(n_t):
                bcs_v_i = full_bcs_v[(i)]
                for bc in bcs_v_i:
                    bc.apply(v.sub(i))

            self._v = v
            self._zeta = zeta

            if space_p is not None:
                p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

                M_p = J_v(p_trial, p_test)
                M_mu = J_u(p_trial, p_test)

                self._M_p = M_p
                self._M_mu = M_mu

                if not CN:
                    flattened_space_p = tuple(space_p for i in range(n_t))
                    mixed_element_p = ufl.classes.MixedElement(
                        *[space.ufl_element() for space in flattened_space_p])
                    full_space_p = FunctionSpace(space_p.mesh(),
                                                 mixed_element_p)
                else:
                    flattened_space_p = tuple(space_p for i in range(n_t - 1))
                    mixed_element_p = ufl.classes.MixedElement(
                        *[space.ufl_element() for space in flattened_space_p])
                    full_space_p = FunctionSpace(space_p.mesh(),
                                                 mixed_element_p)

                p = Function(full_space_p, name="p")
                mu = Function(full_space_p, name="mu")

                self._p = p
                self._mu = mu

        def set_space_v(self, space_v, *, v=None, zeta=None,
                        bcs_v_new=False, bcs_v=None):
            self._space_v = space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            M_v = self._J_v(v_trial, v_test)
            M_zeta = self._J_u(v_trial, v_test)

            self._M_v = M_v
            self._M_zeta = M_zeta

            n_t = self._n_t

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            if v is None:
                v = Function(full_space_v, name="v")
                if self._initial_condition is not None:
                    v_test = TestFunction(space_v)
                    v.sub(0).assign(self._initial_condition(v_test))
            else:
                if v.function_space() != full_space_v:
                    raise ValueError("Unexpected space")

            if bcs_v_new:
                self._f_bcs_v = bcs_v

            full_bcs_v = {}
            if self._f_bcs_v is None:
                for i in range(n_t):
                    full_bcs_v[(i)] = ()
            else:
                t_0 = self._time_interval[0]
                T_f = self._time_interval[1]
                tau = (T_f - t_0) / (n_t - 1.0)
                for i in range(n_t):
                    t = t_0 + i * tau
                    bcs_v_i = self._f_bcs_v(space_v, t)
                    if not isinstance(bcs_v_i, Sequence):
                        full_bcs_v[(i)] = (bcs_v_i, )
                    else:
                        full_bcs_v[(i)] = tuple(bcs_v_i)
            self._bcs_v = full_bcs_v

            for i in range(n_t):
                bcs_v_i = full_bcs_v[(i)]
                for bc in bcs_v_i:
                    bc.apply(v.sub(i))

            if zeta is None:
                zeta = Function(full_space_v, name="zeta")
            else:
                if zeta.function_space() != full_space_v:
                    raise ValueError("Unexpected space")

            bcs_zeta = homogenize(self._bcs_v[(1)])
            for i in range(n_t):
                for bc in bcs_zeta:
                    bc.apply(zeta)

            self._v = v
            self._zeta = zeta

        def set_space_p(self, space_p, *, p=None, mu=None):
            self._space_p = space_p
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            M_p = self._J_v(p_trial, p_test)
            M_mu = self._J_u(p_trial, p_test)

            self._M_p = M_p
            self._M_mu = M_mu

            if not self._CN:
                flattened_space_p = tuple(space_p for i in range(self._n_t))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)
            else:
                flattened_space_p = tuple(
                    space_p for i in range(self._n_t - 1))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

            if p is None:
                p = Function(full_space_p, name="p")
            else:
                if p.function_space() != full_space_p:
                    raise ValueError("Unexpected space")

            if mu is None:
                mu = Function(full_space_p, name="mu")
            else:
                if mu.function_space() != full_space_p:
                    raise ValueError("Unexpected space")

            self._p = p
            self._mu = mu

        def set_forward_form(self, forward_form):
            self._forward_form = forward_form

        def set_desired_state(self, desired_state):
            self._desired_state = desired_state

        def set_force_function(self, force_function):
            self._force_function = force_function

        def set_beta(self, beta):
            self._beta = beta

        def set_initial_condition(self, initial_condition):
            self._initial_condition = initial_condition
            v_test = TestFunction(self._space_v)
            self._v.sub(0).assign(initial_condition(v_test))
            bcs_v_0 = self._bcs_v[(0)]
            for bc in bcs_v_0:
                bc.apply(v.sub(0))

        def set_time_interval(self, time_interval):
            self._time_interval = time_interval

        def set_CN(self, *, CN=True):
            if (self._CN or CN) and not (self._CN and CN):
                self._CN = CN

                if self._space_p is not None:
                    self.set_space_p(self._space_p)

        def set_n_t(self, n_t):
            self._n_t = n_t

            full_bcs_v = {}
            if self._f_bcs_v is None:
                for i in range(n_t):
                    full_bcs_v[(i)] = ()
            else:
                t_0 = self._time_interval[0]
                T_f = self._time_interval[1]
                tau = (T_f - t_0) / (n_t - 1.0)
                for i in range(n_t):
                    t = t_0 + i * tau
                    bcs_v_i = self._f_bcs_v(self._space_v, t)
                    if not isinstance(bcs_v_i, Sequence):
                        full_bcs_v[(i)] = (bcs_v_i, )
                    else:
                        full_bcs_v[(i)] = tuple(bcs_v_i)
            self._bcs_v = full_bcs_v

            flattened_space_v = tuple(self._space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(self._space_v.mesh(), mixed_element_v)

            v = Function(full_space_v, name="v")
            if self._initial_condition is not None:
                v_test = TestFunction(self._space_v)
                v.sub(0).assign(self._initial_condition(v_test))
            for i in range(n_t):
                bcs_v_i = full_bcs_v[(i)]
                for bc in bcs_v_i:
                    bc.apply(v.sub(i))
            zeta = Function(full_space_v, name="zeta")

            self._v = v
            self._zeta = zeta

            if self._space_p is not None:
                if not self._CN:
                    flattened_space_p = tuple(
                        self._space_p for i in range(n_t))
                    mixed_element_p = ufl.classes.MixedElement(
                        *[space.ufl_element() for space in flattened_space_p])
                    full_space_p = FunctionSpace(self._space_p.mesh(),
                                                 mixed_element_p)
                else:
                    flattened_space_p = tuple(
                        self._space_p for i in range(n_t - 1))
                    mixed_element_p = ufl.classes.MixedElement(
                        *[space.ufl_element() for space in flattened_space_p])
                    full_space_p = FunctionSpace(self._space_p.mesh(),
                                                 mixed_element_p)

                p = Function(full_space_p, name="p")
                mu = Function(full_space_p, name="mu")

                self._p = p
                self._mu = mu

        def set_J_v(self, J_v):
            self._J_v = J_v
            v_test = TestFunction(self._space_v)
            v_trial = TrialFunction(self._space_v)
            self._M_v = J_v(v_trial, v_test)

            if self._space_p is not None:
                p_test = TestFunction(self._space_p)
                p_trial = TrialFunction(self._space_p)
                self._M_p = J_v(p_trial, p_test)

        def set_J_u(self, J_u):
            self._J_u = J_u
            v_test = TestFunction(self._space_v)
            v_trial = TrialFunction(self._space_v)
            self._M_zeta = J_u(v_trial, v_test)

            if self._space_p is not None:
                p_test = TestFunction(self._space_p)
                p_trial = TrialFunction(self._space_p)
                self._M_mu = J_u(p_trial, p_test)

        def set_bcs_v(self, bcs_v, space_v=None):
            if space_v is None:
                self._f_bcs_v = bcs_v
                full_bcs_v = {}
                if bcs_v is None:
                    for i in range(self._n_t):
                        full_bcs_v[(i)] = ()
                else:
                    t_0 = self._time_interval[0]
                    T_f = self._time_interval[1]
                    tau = (T_f - t_0) / (self._n_t - 1.0)
                    for i in range(self._n_t):
                        t = t_0 + i * tau
                        bcs_v_i = bcs_v(self._space_v, t)
                        if not isinstance(bcs_v_i, Sequence):
                            full_bcs_v[(i)] = (bcs_v_i, )
                        else:
                            full_bcs_v[(i)] = tuple(bcs_v_i)
                self._bcs_v = full_bcs_v

                for i in range(self._n_t):
                    bcs_v_i = full_bcs_v[(i)]
                    for bc in bcs_v_i:
                        bc.apply(self._v.sub(i))

                bcs_zeta = homogenize(self._bcs_v[(1)])
                for i in range(n_t):
                    for bc in bcs_zeta:
                        bc.apply(zeta)
            else:
                self._set_space_v(space_v, bcs_v_new=True, bcs_v=bcs_v)

        def set_Gauss_Newton(self, Gauss_Newton=True):
            self._Gauss_Newton = Gauss_Newton

        def set_v(self, v_new):
            if v_new.function_space() != self._v.function_space():
                raise ValueError("Unexpected space")
            else:
                self._v.assign(v_new)
                for i in range(self._n_t):
                    bcs_v_i = self._bcs_v[(i)]
                    for bc in bcs_v_i:
                        bc.apply(self._v.sub(i))

        def set_zeta(self, zeta_new):
            if zeta_new.function_space() != self._zeta.function_space():
                raise ValueError("Unexpected space")
            else:
                self._zeta.assign(zeta_new)
                bcs_zeta = homogenize(self._bcs_v[(1)])
                for i in range(self._n_t):
                    for bc in bcs_zeta:
                        bc.apply(self._zeta.sub(i))

        def set_p(self, p_new):
            if self._space_p is not None:
                if p_new.function_space() != self._p.function_space():
                    raise ValueError("Unexpected space")
                else:
                    self._p.assign(p_new)
            else:
                raise ValueError("Undefined space_p: unable to assign value")

        def set_mu(self, mu_new):
            if self._space_p is not None:
                if mu_new.function_space() != self._mu.function_space():
                    raise ValueError("Unexpected space")
                else:
                    self._mu.assign(mu_new)
            else:
                raise ValueError("Undefined space_p: unable to assign value")

        def print_error(self, full_space_v, v_test):
            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]

            tau = (T_f - t_0) / (n_t - 1.0)

            true_v = Function(full_space_v, name="true_v")
            for i in range(n_t):
                t = t_0 + i * tau
                v_d_i, true_v_i = self._desired_state(v_test, t)
                true_v.sub(i).assign(true_v_i)
            v_err = true_v - self._v
            error = sqrt(tau) * sqrt(abs(assemble(
                inner(v_err, v_err) * dx)))
            print('estimated error in the L2-norm: ', error)

        def construct_D_v(self, v_trial, v_test, v_n_help, t):
            if not self._Gauss_Newton:
                D_v_i = self._forward_form(v_trial, v_test, v_n_help, t)
            else:
                D_v_i = ufl.derivative(
                    self._forward_form(v_trial, v_test, v_n_help, t),
                    v_n_help,
                    v_trial)

            return D_v_i

        def construct_f(self, full_space_v, v_test):
            f = Cofunction(full_space_v.dual(), name="f")

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]

            tau = (T_f - t_0) / (n_t - 1.0)

            for i in range(n_t):
                t = t_0 + i * tau
                f.sub(i).assign(assemble(self._force_function(v_test, t)))

            return f

        def construct_v_d(self, full_space_v, v_test):
            v_d = Cofunction(full_space_v.dual(), name="v_d")
            true_v = Function(full_space_v, name="true_v")

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]

            tau = (T_f - t_0) / (n_t - 1.0)

            for i in range(n_t):
                t = t_0 + i * tau
                v_d_i, true_v_i = self._desired_state(v_test, t)
                v_d.sub(i).assign(assemble(v_d_i))
                true_v.sub(i).assign(true_v_i)

            return v_d, true_v

        def construct_pc(self, Multigrid, lambda_v_bounds, full_space_v,
                         bcs_v, bcs_zeta, block_01, block_10, epsilon):
            space_v = self._space_v
            n_t = self._n_t
            beta = self._beta
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]

            tau = (T_f - t_0) / (n_t - 1.0)

            # definition of preconditioner
            if self._CN:
                def pc_linear(u_0, u_1, b_0, b_1):
                    # solving for the (1,1)-block
                    if Multigrid:
                        solver_0 = LinearSolver(
                            assemble(self._M_v,
                                     bcs=bcs_v),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                    else:
                        if lambda_v_bounds is not None:
                            e_min = lambda_v_bounds[0]
                            e_max = lambda_v_bounds[1]
                            solver_0 = LinearSolver(
                                assemble(self._M_v,
                                         bcs=bcs_v),
                                solver_parameters={
                                    "ksp_type": "chebyshev",
                                    "pc_type": "jacobi",
                                    "ksp_chebyshev_eigenvalues": f"{e_min:.16e}, {e_max:.16e}",  # noqa: E501
                                    "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                                    "ksp_chebyshev_esteig_steps": 0,
                                    "ksp_chebyshev_esteig_noisy": False,
                                    "ksp_max_it": 20,
                                    "ksp_atol": 0.0,
                                    "ksp_rtol": 0.0})
                        else:
                            solver_0 = LinearSolver(
                                assemble(self._M_v,
                                         bcs=bcs_v),
                                solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "jacobi",
                                                   "ksp_max_it": 1,
                                                   "ksp_atol": 0.0,
                                                   "ksp_rtol": 0.0})

                    b_0_help = apply_T_1_inv(b_0, space_v, n_t - 1)

                    for i in range(n_t - 1):
                        b = Cofunction(space_v.dual())
                        b.assign(b_0_help.sub(i))
                        try:
                            solver_0.solve(u_0.sub(i),
                                           b.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_0.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b
                        with u_0.sub(i).dat.vec as x_v:
                            x_v.scale(2.0 / tau)

                    b_0_help = apply_T_2_inv(u_0, space_v, n_t - 1)
                    for i in range(n_t - 1):
                        u_0.sub(i).assign(b_0_help.sub(i))
                    del b_0_help

                    # u_1 = - b_1 + D_v * u_0
                    b = Cofunction(full_space_v.dual())
                    block_ii = block_10[(0, 0)]
                    b_help = Function(space_v)
                    b_help.assign(u_0.sub(0))
                    b.sub(0).assign(assemble(action(block_ii, b_help)))
                    for bc in bcs_zeta:
                        bc.apply(b.sub(0))
                    del b_help
                    for i in range(1, n_t - 1):
                        block_ij = block_10[(i, i - 1)]
                        b_help = Function(space_v)
                        b_help.assign(u_0.sub(i - 1))
                        b_help_new = assemble(action(block_ij, b_help))
                        block_ii = block_10[(i, i)]
                        b_help.assign(u_0.sub(i))
                        b.sub(i).assign(assemble(action(block_ii, b_help)))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(1.0, b_1_v)
                        del b_help
                        del b_help_new
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))

                    b = apply_T_2(b, space_v, n_t - 1)

                    for i in range(n_t - 1):
                        with b.sub(i).dat.vec as b_v, \
                                b_1.sub(i).dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))

                    # solving for the Schur complement approximation
                    b = apply_T_2_inv(b, space_v, n_t - 1)
                    # first solve
                    block_ii = block_10[(0, 0)]
                    solver_1 = LinearSolver(
                        assemble(block_ii
                                 + (0.5 * tau / (beta**0.5)) * self._M_v,
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(0))
                    try:
                        solver_1.solve(u_1.sub(0),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help

                    for i in range(1, n_t - 1):
                        block_ij = block_10[(i, i - 1)]
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i - 1))
                        b_help_new = assemble(action(block_ij, b_help))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        b_help_new = assemble(
                            action((0.5 * tau / (beta**0.5)) * self._M_v,
                                   b_help))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del b_help_new
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))
                        b_help = Cofunction(space_v.dual())
                        b_help.assign(b.sub(i))

                        block_ii = block_10[(i, i)]
                        solver_1 = LinearSolver(
                            assemble(block_ii
                                     + (0.5 * tau / beta**0.5) * self._M_v,
                                     bcs=bcs_zeta),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                        try:
                            solver_1.solve(u_1.sub(i),
                                           b_help.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b_help
                    del b

                    b = apply_T_2(u_1, space_v, n_t - 1)
                    for i in range(n_t - 1):
                        u_1.sub(i).assign(b.sub(i))
                    del b

                    b = Cofunction(full_space_v.dual())
                    for i in range(n_t - 1):
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i))
                        b.sub(i).assign(assemble(action(self._M_v,
                                                        b_help)))
                        del b_help
                        with b.sub(i).dat.vec as b_v:
                            b_v.scale(0.5 * tau)
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))

                    # second solve
                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(n_t - 2))
                    block_ii = block_01[(n_t - 2, n_t - 2)]
                    solver_2 = LinearSolver(
                        assemble(block_ii
                                 + (0.5 * tau / beta**0.5) * self._M_zeta,
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                    try:
                        solver_2.solve(u_1.sub(n_t - 2),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help

                    for i in range(n_t - 3, -1, -1):
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i + 1))
                        block_ij = block_01[(i, i + 1)] + (0.5 * tau / beta**0.5) * self._M_zeta  # noqa: E501
                        b_help_new = assemble(action(block_ij, b_help))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del b_help_new
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))
                        b_help = Cofunction(space_v.dual())
                        b_help.assign(b.sub(i))
                        block_ii = block_01[(i, i)]
                        solver_2 = LinearSolver(
                            assemble(block_ii
                                     + (0.5 * tau / beta**0.5) * self._M_zeta,
                                     bcs=bcs_zeta),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                        try:
                            solver_2.solve(u_1.sub(i),
                                           b_help.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b_help
            else:
                def pc_linear(u_0, u_1, b_0, b_1):
                    # solving for the (1,1)-block
                    if Multigrid:
                        solver_0 = LinearSolver(
                            assemble(self._M_v,
                                     bcs=bcs_v),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                    else:
                        if lambda_v_bounds is not None:
                            e_min = lambda_v_bounds[0]
                            e_max = lambda_v_bounds[1]
                            solver_0 = LinearSolver(
                                assemble(self._M_v,
                                         bcs=bcs_v),
                                solver_parameters={
                                    "ksp_type": "chebyshev",
                                    "pc_type": "jacobi",
                                    "ksp_chebyshev_eigenvalues": f"{e_min:.16e}, {e_max:.16e}",  # noqa: E501
                                    "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                                    "ksp_chebyshev_esteig_steps": 0,
                                    "ksp_chebyshev_esteig_noisy": False,
                                    "ksp_max_it": 20,
                                    "ksp_atol": 0.0,
                                    "ksp_rtol": 0.0})
                        else:
                            solver_0 = LinearSolver(
                                assemble(self._M_v,
                                         bcs=bcs_v),
                                solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "jacobi",
                                                   "ksp_max_it": 1,
                                                   "ksp_atol": 0.0,
                                                   "ksp_rtol": 0.0})

                    for i in range(n_t):
                        b = Cofunction(space_v.dual())
                        b.assign(b_0.sub(i))
                        try:
                            solver_0.solve(
                                u_0.sub(i), b.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_0.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b
                        with u_0.sub(i).dat.vec as x_v:
                            x_v.scale(1.0 / tau)

                    with u_0.sub(n_t - 1).dat.vec as x_v:
                        x_v.scale(1.0 / epsilon)

                    # u_1 = - b_1 + D_v * u_0
                    b = Cofunction(full_space_v.dual())
                    block_ii = block_10[(0, 0)]
                    b_help = Function(space_v)
                    b_help.assign(u_0.sub(0))
                    b.sub(0).assign(assemble(action(block_ii, b_help)))
                    with b.sub(0).dat.vec as b_v, \
                            b_1.sub(0).dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_zeta:
                        bc.apply(b.sub(0))
                    del b_help
                    for i in range(1, n_t):
                        block_ij = block_10[(i, i - 1)]
                        b_help = Function(space_v)
                        b_help.assign(u_0.sub(i - 1))
                        b_help_new = assemble(action(block_ij, b_help))
                        block_ii = block_10[(i, i)]
                        b_help.assign(u_0.sub(i))
                        b.sub(i).assign(assemble(action(block_ii, b_help)))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(1.0, b_1_v)
                        del b_help
                        del b_help_new
                        with b.sub(i).dat.vec as b_v, \
                                b_1.sub(i).dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))

                    # solving for the Schur complement approximation
                    # first solve
                    block_ii = block_10[(0, 0)]
                    solver_1 = LinearSolver(
                        assemble(block_ii,
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})

                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(0))
                    try:
                        solver_1.solve(u_1.sub(0),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help

                    for i in range(1, n_t - 1):
                        block_ij = block_10[(i, i - 1)]
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i - 1))
                        b_help_new = assemble(action(block_ij, b_help))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del b_help_new
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))
                        b_help = Cofunction(space_v.dual())
                        b_help.assign(b.sub(i))
                        block_ii = block_10[(i, i)]
                        solver_1 = LinearSolver(
                            assemble(block_ii
                                     + (tau / (beta**0.5)) * self._M_v,
                                     bcs=bcs_zeta),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})

                        try:
                            solver_1.solve(u_1.sub(i),
                                           b_help.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b_help

                    block_ij = block_10[(n_t - 1, n_t - 2)]
                    b_help = Function(space_v)
                    b_help.assign(u_1.sub(n_t - 2))
                    b_help_new = assemble(action(block_ij, b_help))
                    with b.sub(n_t - 1).dat.vec as b_v, \
                            b_help_new.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_zeta:
                        bc.apply(b.sub(n_t - 1))
                    del b_help_new
                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(n_t - 1))
                    block_ii = block_10[(n_t - 1, n_t - 1)]
                    solver_1 = LinearSolver(
                        assemble(block_ii
                                 + ((epsilon**0.5) * tau / (beta**0.5)) * self._M_v,  # noqa: E501
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                    try:
                        solver_1.solve(u_1.sub(n_t - 1),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help
                    del b

                    b = Cofunction(full_space_v.dual())
                    for i in range(n_t - 1):
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i))
                        b.sub(i).assign(assemble(action(self._M_v,
                                                        b_help)))
                        del b_help
                        with b.sub(i).dat.vec as b_v:
                            b_v.scale(tau)
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))

                    b_help = Function(space_v)
                    b_help.assign(u_1.sub(n_t - 1))
                    b.sub(n_t - 1).assign(assemble(action(self._M_v,
                                                          b_help)))
                    del b_help
                    with b.sub(n_t - 1).dat.vec as b_v:
                        b_v.scale(epsilon * tau)
                    for bc in bcs_zeta:
                        bc.apply(b.sub(n_t - 1))

                    # second solve
                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(n_t - 1))
                    block_ii = block_01[(n_t - 1, n_t - 1)]
                    solver_2 = LinearSolver(
                        assemble(block_ii
                                 + ((epsilon**0.5) * tau / (beta**0.5)) * self._M_zeta,  # noqa: E501
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                    try:
                        solver_2.solve(u_1.sub(n_t - 1),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help

                    for i in range(n_t - 2, 0, -1):
                        b_help = Function(space_v)
                        b_help.assign(u_1.sub(i + 1))
                        block_ij = block_01[(i, i + 1)]
                        b_help_new = assemble(action(block_ij, b_help))
                        with b.sub(i).dat.vec as b_v, \
                                b_help_new.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del b_help_new
                        for bc in bcs_zeta:
                            bc.apply(b.sub(i))
                        b_help = Cofunction(space_v.dual())
                        b_help.assign(b.sub(i))
                        block_ii = block_01[(i, i)]
                        solver_2 = LinearSolver(
                            assemble(block_ii
                                     + (tau / (beta**0.5)) * self._M_zeta,
                                     bcs=bcs_zeta),
                            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "hypre",
                                "pc_hypre_type": "boomeramg",
                                "ksp_max_it": 1,
                                "pc_hypre_boomeramg_max_iter": 2,
                                "ksp_atol": 0.0,
                                "ksp_rtol": 0.0})
                        try:
                            solver_2.solve(u_1.sub(i),
                                           b_help.copy(deepcopy=True))
                        except ConvergenceError:
                            assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del b_help

                    b_help = Function(space_v)
                    b_help.assign(u_1.sub(1))
                    block_ij = block_01[(0, 1)]
                    b_help_new = assemble(action(block_ij, b_help))
                    with b.sub(0).dat.vec as b_v, \
                            b_help_new.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    del b_help_new
                    for bc in bcs_zeta:
                        bc.apply(b.sub(0))
                    b_help = Cofunction(space_v.dual())
                    b_help.assign(b.sub(0))
                    block_ii = block_01[(0, 0)]
                    solver_2 = LinearSolver(
                        assemble(block_ii,
                                 bcs=bcs_zeta),
                        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_max_it": 1,
                            "pc_hypre_boomeramg_max_iter": 2,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                    try:
                        solver_2.solve(u_1.sub(0),
                                       b_help.copy(deepcopy=True))
                    except ConvergenceError:
                        assert solver_2.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                    del b_help
                    del b

            return pc_linear

        def non_linear_res_eval(self, full_space_v, v_old, zeta_old, v_0,
                                v_d, f, M_v, bcs_v, bcs_zeta):
            space_v = self._space_v
            n_t = self._n_t
            beta = self._beta
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]

            tau = (T_f - t_0) / (n_t - 1.0)

            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            if not self._CN:
                rhs_0 = Cofunction(full_space_v.dual(), name="rhs_0")
                rhs_1 = Cofunction(full_space_v.dual(), name="rhs_1")

                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_old.sub(0), t_0)
                D_zeta_i = adjoint(D_v_i)

                D_v_0 = self.construct_D_v(v_trial, v_test, v_0, t_0)

                rhs_0.sub(0).assign(tau * v_d.sub(0))
                b_help = Function(space_v)
                b_help.assign(v_old.sub(0))
                b = assemble(action(tau * self._M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(0))
                b = assemble(action(tau * D_zeta_i + M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(1))
                b = assemble(action(M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(1.0, b_v)
                del b
                del b_help
                for bc in bcs_zeta:
                    bc.apply(rhs_0.sub(0))

                b = assemble(action(tau * D_v_0 + M_v, v_0))
                rhs_1.sub(0).assign(b)
                del b

                b_help = Function(space_v)
                b_help.assign(v_old.sub(0))
                b = assemble(action(tau * D_v_i + M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(0).dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b
                del b_help
                for bc in bcs_v:
                    bc.apply(rhs_1.sub(0))

                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_old.sub(n_t - 1), T_f)

                rhs_1.sub(n_t - 1).assign(tau * f.sub(n_t - 1))
                b_help = Function(space_v)
                b_help.assign(v_old.sub(n_t - 2))
                b = assemble(action(M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(n_t - 1).dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(v_old.sub(n_t - 1))
                b = assemble(action(tau * D_v_i + M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(n_t - 1).dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(n_t - 1))
                b = assemble(action((tau / beta) * self._M_zeta, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(n_t - 1).dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                del b
                del b_help
                for bc in bcs_v:
                    bc.apply(rhs_1.sub(n_t - 1))

                for i in range(1, n_t - 1):
                    t = t_0 + i * tau
                    D_v_i = self.construct_D_v(v_trial, v_test,
                                               v_old.sub(i), t)
                    D_zeta_i = adjoint(D_v_i)

                    rhs_0.sub(i).assign(tau * v_d.sub(i))
                    rhs_1.sub(i).assign(tau * f.sub(i))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i))
                    b = assemble(action(tau * self._M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i))
                    b = assemble(action(tau * D_zeta_i + M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i + 1))
                    b = assemble(action(M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(1.0, b_v)
                    del b
                    del b_help
                    for bc in bcs_zeta:
                        bc.apply(rhs_0.sub(i))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i))
                    b = assemble(action(tau * D_v_i + M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i - 1))
                    b = assemble(action(M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i))
                    b = assemble(action((tau / beta) * self._M_zeta,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(1.0, b_v)
                    del b
                    del b_help
                    for bc in bcs_v:
                        bc.apply(rhs_1.sub(i))
            else:
                rhs_0 = Cofunction(full_space_v.dual(), name="rhs_0")
                rhs_1 = Cofunction(full_space_v.dual(), name="rhs_1")

                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_old.sub(0), t_0)
                D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                v_old.sub(1), t_0 + tau)
                D_zeta_i = adjoint(D_v_i)
                D_zeta_i_plus = adjoint(D_v_i_plus)

                rhs_0.sub(0).assign(0.5 * tau * (v_d.sub(0) + v_d.sub(1)))
                b_help = Function(space_v)
                b_help.assign(v_old.sub(0))
                b = assemble(action(0.5 * tau * self._M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b

                b_help = Function(space_v)
                b_help.assign(v_old.sub(1))
                b = assemble(action(0.5 * tau * self._M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(0))
                b = assemble(action(0.5 * tau * D_zeta_i + M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(1))
                b = assemble(action(0.5 * tau * D_zeta_i_plus - M_v,
                                    b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_0.sub(0).dat.vec as b_0_v:
                    b_0_v.axpy(-1.0, b_v)
                del b
                del b_help
                for bc in bcs_zeta:
                    bc.apply(rhs_0.sub(0))

                rhs_1.sub(0).assign(0.5 * tau * (f.sub(0) + f.sub(1)))
                b_help = Function(space_v)
                b_help.assign(v_old.sub(0))
                b = assemble(action(0.5 * tau * D_v_i - M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(0).dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(v_old.sub(1))
                b = assemble(action(0.5 * tau * D_v_i_plus + M_v, b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(0).dat.vec as b_1_v:
                    b_1_v.axpy(-1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(0))
                b = assemble(action(0.5 * (tau / beta) * self._M_zeta,
                                    b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(0).dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                del b
                del b_help

                b_help = Function(space_v)
                b_help.assign(zeta_old.sub(1))
                b = assemble(action(0.5 * (tau / beta) * self._M_zeta,
                                    b_help))
                with b.dat.vec_ro as b_v, \
                        rhs_1.sub(0).dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                del b
                del b_help
                for bc in bcs_v:
                    bc.apply(rhs_1.sub(0))

                for i in range(1, n_t - 1):
                    t = t_0 + i * tau
                    D_v_i = self.construct_D_v(v_trial, v_test,
                                               v_old.sub(i), t)
                    D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                    v_old.sub(i + 1), t + tau)
                    D_zeta_i = adjoint(D_v_i)
                    D_zeta_i_plus = adjoint(D_v_i_plus)

                    rhs_0.sub(i).assign(
                        0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1)))
                    rhs_1.sub(i).assign(
                        0.5 * tau * (f.sub(i) + f.sub(i + 1)))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i))
                    b = assemble(action(0.5 * tau * self._M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i + 1))
                    b = assemble(action(0.5 * tau * self._M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i))
                    b = assemble(action(0.5 * tau * D_zeta_i + M_v,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i + 1))
                    b = assemble(action(0.5 * tau * D_zeta_i_plus - M_v,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_0.sub(i).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_help
                    for bc in bcs_zeta:
                        bc.apply(rhs_0.sub(i))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i))
                    b = assemble(action(0.5 * tau * D_v_i - M_v, b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(i + 1))
                    b = assemble(action(0.5 * tau * D_v_i_plus + M_v,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(-1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i))
                    b = assemble(action(0.5 * (tau / beta) * self._M_zeta,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(1.0, b_v)
                    del b
                    del b_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(i + 1))
                    b = assemble(action(0.5 * (tau / beta) * self._M_zeta,
                                        b_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_1.sub(i).dat.vec as b_1_v:
                        b_1_v.axpy(1.0, b_v)
                    del b
                    del b_help
                    for bc in bcs_v:
                        bc.apply(rhs_1.sub(i))

            return rhs_0, rhs_1

        def linear_solve(self, *,
                         P=None, solver_parameters=None, Multigrid=False,
                         lambda_v_bounds=None, v_d=None, f=None,
                         print_error=True, create_output=True,
                         plots=False):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]
            tau = (T_f - t_0) / (n_t - 1.0)

            beta = self._beta

            if self._CN:
                epsilon = None
            else:
                epsilon = 10.0**-3

            inhomogeneous_bcs_v = False
            for (i), bc_i in self._bcs_v.items():
                for bc in bc_i:
                    if not isinstance(bc.function_arg, ufl.classes.Zero):
                        inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v[(1)])
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v[(1)]
            bcs_zeta = bcs_v

            nullspace_v = DirichletBCNullspace(bcs_v)
            nullspace_zeta = DirichletBCNullspace(bcs_zeta)

            full_nullspace_v = ()
            full_nullspace_zeta = ()
            for i in range(n_t - 1):
                full_nullspace_v = full_nullspace_v + (nullspace_v, )
                full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )

            if not self._CN:
                full_nullspace_v = full_nullspace_v + (nullspace_v, )
                full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            if self._initial_condition is not None:
                v_0 = self._initial_condition(v_test)
            else:
                v_0 = Function(space_v, name="v_0")

            if f is None:
                check_f = True
                f = self.construct_f(full_space_v, v_test)
            else:
                check_f = False

            if v_d is None:
                check_v_d = True
                v_d, true_v = self.construct_v_d(full_space_v, v_test)
            else:
                check_v_d = False
                true_v = Function(full_space_v, name="true_v")
                for i in range(n_t):
                    t = t_0 + i * tau
                    v_d_i, true_v_i = self._desired_state(v_test, t)
                    true_v.sub(i).assign(true_v_i)

            v_old = self._v
            v_n_help = Function(space_v)

            M_v = inner(v_trial, v_test) * dx

            block_00 = {}
            block_01 = {}
            block_10 = {}
            block_11 = {}

            if not self._CN:
                for j in range(n_t):
                    block_11[(0, j)] = None

            for i in range(n_t - 1):
                t = t_0 + i * tau
                v_n_help.assign(v_old.sub(i))

                D_v_i = self.construct_D_v(v_trial, v_test, v_n_help, t)
                D_zeta_i = adjoint(D_v_i)

                if not self._CN:
                    for j in range(n_t):
                        if j == i - 1:
                            block_00[(i, j)] = None
                            block_01[(i, j)] = None
                            block_10[(i, j)] = -M_v
                            block_11[(i + 1, j)] = None
                        elif j == i:
                            block_00[(i, j)] = tau * self._M_v
                            block_01[(i, j)] = tau * D_zeta_i + M_v
                            block_10[(i, j)] = tau * D_v_i + M_v
                            block_11[(i + 1, j)] = None
                        elif j == i + 1:
                            block_00[(i, j)] = None
                            block_01[(i, j)] = -M_v
                            block_10[(i, j)] = None
                            block_11[(i + 1, j)] = - (tau / beta) * self._M_zeta  # noqa: E501
                        else:
                            block_00[(i, j)] = None
                            block_01[(i, j)] = None
                            block_10[(i, j)] = None
                            block_11[(i + 1, j)] = None
                else:
                    v_n_help_plus = Function(space_v)
                    v_n_help_plus.assign(v_old.sub(i + 1))

                    D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                    v_n_help_plus, t + tau)
                    D_zeta_i_plus = adjoint(D_v_i_plus)

                    for j in range(n_t - 1):
                        if j == i - 1:
                            block_00[(i, j)] = 0.5 * tau * self._M_v
                            block_01[(i, j)] = None
                            block_10[(i, j)] = 0.5 * tau * D_v_i - M_v
                            block_11[(i, j)] = None
                        elif j == i:
                            block_00[(i, j)] = 0.5 * tau * self._M_v
                            block_01[(i, j)] = 0.5 * tau * D_zeta_i_plus + M_v
                            block_10[(i, j)] = 0.5 * tau * D_v_i_plus + M_v
                            block_11[(i, j)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501
                        elif j == i + 1:
                            block_00[(i, j)] = None
                            block_01[(i, j)] = 0.5 * tau * D_zeta_i - M_v
                            block_10[(i, j)] = None
                            block_11[(i, j)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501
                        else:
                            block_00[(i, j)] = None
                            block_01[(i, j)] = None
                            block_10[(i, j)] = None
                            block_11[(i, j)] = None

            if not self._CN:
                t = T_f
                v_n_help.assign(v_old.sub(n_t - 1))

                D_v_i = self.construct_D_v(v_trial, v_test, v_n_help, t)
                D_zeta_i = adjoint(D_v_i)

                for j in range(n_t - 2):
                    block_00[(n_t - 1, j)] = None
                    block_01[(n_t - 1, j)] = None
                    block_10[(n_t - 1, j)] = None

                block_00[(n_t - 1, n_t - 2)] = None
                block_00[(n_t - 1, n_t - 1)] = None
                block_01[(n_t - 1, n_t - 2)] = None
                block_01[(n_t - 1, n_t - 1)] = tau * D_zeta_i + M_v
                block_10[(n_t - 1, n_t - 2)] = - M_v
                block_10[(n_t - 1, n_t - 1)] = tau * D_v_i + M_v

            if not self._CN:
                b_0 = Cofunction(full_space_v.dual(), name="b_0")
                b_1 = Cofunction(full_space_v.dual(), name="b_1")
            else:
                flattened_space_v_help = tuple(space_v for i in range(n_t - 1))
                mixed_element_v_help = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_v_help])
                full_space_v_help = FunctionSpace(space_v.mesh(),
                                                  mixed_element_v_help)

                b_0 = Cofunction(full_space_v_help.dual(), name="b_0")
                b_1 = Cofunction(full_space_v_help.dual(), name="b_1")

            if not self._CN:
                if check_v_d:
                    b_0.sub(0).assign(tau * v_d.sub(0))

                    if inhomogeneous_bcs_v:
                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(0)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * self._M_v, v_inhom))
                        with b_0.sub(0).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del v_inhom
                        del b_help

                    for bc in bcs_zeta:
                        bc.apply(b_0.sub(0))
                else:
                    b_0.sub(0).assign(v_d.sub(0))

                if check_f:
                    D_v_i = self.construct_D_v(v_trial, v_test, v_0, t_0)

                    b_1.sub(0).assign(assemble(action(tau * D_v_i + M_v,
                                                      v_0)))

                    if inhomogeneous_bcs_v:
                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(0)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * D_v_i + M_v, v_inhom))
                        with b_1.sub(0).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del v_inhom
                        del b_help

                    for bc in bcs_v:
                        bc.apply(b_1.sub(0))
                else:
                    b_1.sub(0).assign(f.sub(0))

                for i in range(1, n_t - 1):
                    if check_v_d:
                        b_0.sub(i).assign(tau * v_d.sub(i))

                        if inhomogeneous_bcs_v:
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * self._M_v, v_inhom))
                            with b_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help

                        for bc in bcs_zeta:
                            bc.apply(b_0.sub(i))
                    else:
                        b_0.sub(i).assign(v_d.sub(i))

                    if check_f:
                        b_1.sub(i).assign(tau * f.sub(i))

                        if inhomogeneous_bcs_v:
                            t = t_0 + i * tau
                            v_n_help.assign(v_old.sub(i))

                            D_v_i = self.construct_D_v(v_trial, v_test,
                                                       v_n_help, t)

                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * D_v_i + M_v,
                                                     v_inhom))
                            with b_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                            del v_inhom
                            del b_help

                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i - 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(M_v, v_inhom))
                            with b_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(1.0, b_1_v)

                            del v_inhom
                            del b_help

                        for bc in bcs_v:
                            bc.apply(b_1.sub(i))
                    else:
                        b_1.sub(i).assign(f.sub(i))

                if not check_v_d:
                    b_0.sub(n_t - 1).assign(v_d.sub(n_t - 1))

                if check_f:
                    b_1.sub(n_t - 1).assign(tau * f.sub(n_t - 1))
                    if inhomogeneous_bcs_v:
                        v_n_help.assign(v_old.sub(n_t - 1))

                        D_v_i = self.construct_D_v(v_trial, v_test,
                                                   v_n_help, T_f)

                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(n_t - 1)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * D_v_i + M_v, v_inhom))
                        with b_1.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)

                        del v_inhom
                        del b_help

                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(n_t - 2)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(M_v, v_inhom))
                        with b_1.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(1.0, b_1_v)

                        del v_inhom
                        del b_help

                    for bc in bcs_v:
                        bc.apply(b_1.sub(n_t - 1))
                else:
                    b_1.sub(n_t - 1).assign(f.sub(n_t - 1))
            else:
                for i in range(n_t - 1):
                    if check_v_d:
                        b_0.sub(i).assign(
                            0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1)))

                        if inhomogeneous_bcs_v:
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i + 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(0.5 * tau * self._M_v,
                                                     v_inhom))
                            with b_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                            del v_inhom
                            del b_help

                            if i > 0:
                                v_inhom = Function(space_v)

                                for bc in bcs_v_help[(i)]:
                                    bc.apply(v_inhom)

                                b_help = assemble(action(0.5 * tau * self._M_v,
                                                         v_inhom))

                                with b_0.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(-1.0, b_1_v)

                                del v_inhom
                                del b_help

                        for bc in bcs_zeta:
                            bc.apply(b_0.sub(i))
                    else:
                        b_0.sub(i).assign(v_d.sub(i))

                    if check_f:
                        b_1.sub(i).assign(
                            0.5 * tau * (f.sub(i) + f.sub(i + 1)))

                        if inhomogeneous_bcs_v:
                            t = t_0 + (i + 1) * tau
                            v_n_help.assign(v_old.sub(i + 1))

                            D_v_i = self.construct_D_v(v_trial, v_test,
                                                       v_n_help, t)

                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i + 1)]:
                                bc.apply(v_inhom)

                            b_help = assemble(action(0.5 * tau * D_v_i + M_v,
                                                     v_inhom))
                            with b_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                            del v_inhom
                            del b_help

                            if i > 0:
                                t = t_0 + i * tau
                                v_n_help.assign(v_old.sub(i))

                                D_v_i = self.construct_D_v(v_trial, v_test,
                                                           v_n_help, t)

                                v_inhom = Function(space_v)
                                for bc in bcs_v_help[(i)]:
                                    bc.apply(v_inhom)
                                b_help = assemble(
                                    action(0.5 * tau * D_v_i - M_v, v_inhom))
                                with b_1.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(-1.0, b_1_v)

                                del v_inhom
                                del b_help
                        for bc in bcs_v:
                            bc.apply(b_1.sub(i))
                    else:
                        b_1.sub(i).assign(f.sub(i))

                if check_v_d:
                    b = assemble(action(0.5 * tau * self._M_v, v_0))
                    with b_0.sub(0).dat.vec as b_v, \
                            b.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_zeta:
                        bc.apply(b_0.sub(0))

                    del b

                if check_f:
                    D_v_i = self.construct_D_v(v_trial, v_test,
                                               v_0, t_0)

                    b = assemble(action(0.5 * tau * D_v_i - M_v, v_0))
                    with b_1.sub(0).dat.vec as b_v, \
                            b.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_v:
                        bc.apply(b_1.sub(0))

                    del b

                b_0 = apply_T_1(b_0, space_v, n_t - 1)
                b_1 = apply_T_2(b_1, space_v, n_t - 1)

            if P is None:
                if self._CN:
                    pc_fn = self.construct_pc(Multigrid, lambda_v_bounds,
                                              full_space_v_help,
                                              bcs_v, bcs_zeta,
                                              block_01, block_10, epsilon)
                else:
                    pc_fn = self.construct_pc(Multigrid, lambda_v_bounds,
                                              full_space_v,
                                              bcs_v, bcs_zeta,
                                              block_01, block_10, epsilon)
            else:
                pc_fn = P

            if solver_parameters is None:
                solver_parameters = {"linear_solver": "gmres",
                                     "gmres_restart": 10,
                                     "maximum_iterations": 50,
                                     "relative_tolerance": 1.0e-6,
                                     "absolute_tolerance": 0.0,
                                     "monitor_convergence": print_error}

            if not self._CN:
                system = MultiBlockSystem(
                    space_v, space_v,
                    block_00=block_00, block_01=block_01,
                    block_10=block_10, block_11=block_11,
                    n_blocks_00=n_t, n_blocks_11=n_t,
                    nullspace_0=full_nullspace_v,
                    nullspace_1=full_nullspace_zeta,
                    CN=self._CN)
            else:
                system = MultiBlockSystem(
                    space_v, space_v,
                    block_00=block_00, block_01=block_01,
                    block_10=block_10, block_11=block_11,
                    n_blocks_00=n_t - 1, n_blocks_11=n_t - 1,
                    nullspace_0=full_nullspace_v,
                    nullspace_1=full_nullspace_zeta,
                    CN=self._CN)

            if not self._CN:
                v = Function(full_space_v, name="v")
                zeta = Function(full_space_v, name="zeta")
            else:
                v = Function(full_space_v_help, name="v")
                zeta = Function(full_space_v_help, name="zeta")

            system.solve(
                v, zeta, b_0, b_1,
                solver_parameters=solver_parameters,
                pc_fn=pc_fn)

            if self._CN:
                v_new = Function(full_space_v, name="v_new")
                zeta_new = Function(full_space_v, name="zeta_new")

                if check_f and check_v_d:
                    v_new.sub(0).assign(v_0)

                for i in range(n_t - 1):
                    v_new.sub(i + 1).assign(v.sub(i))
                    zeta_new.sub(i).assign(zeta.sub(i))

                self.set_v(v_new)
                self.set_zeta(zeta_new)
            else:
                if check_f and check_v_d:
                    v.sub(0).assign(v_0)

                self.set_v(v)
                self.set_zeta(zeta)

            if print_error:
                self.print_error(full_space_v, v_test)

            if create_output:
                v_output = File("v.pvd")
                zeta_output = File("zeta.pvd")

                b_help_v = Function(space_v, name="b_help_v")

                for i in range(n_t):
                    t = t_0 + i * tau

                    b_help_v.interpolate(v.sub(i))
                    v_output.write(b_help_v, t=t)

                    b_help_v.interpolate(zeta.sub(i))
                    zeta_output.write(b_help_v, t=t)

                del b_help_v

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(v)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(zeta)

            if plots:
                for i in range(n_t):
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(v.sub(i), axes=axes)
                        fig_v.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(zeta.sub(i), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(i), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)

        def non_linear_solve(self, *,
                             P=None, solver_parameters=None,
                             Multigrid=False,
                             lambda_v_bounds=None,
                             max_non_linear_iter=10,
                             relative_non_linear_tol=10.0**-5,
                             absolute_non_linear_tol=10.0**-8,
                             print_error_linear=False,
                             print_error_non_linear=True,
                             create_output=True,
                             plots=False):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]
            tau = (T_f - t_0) / (n_t - 1.0)

            inhomogeneous_bcs_v = False
            for (i), bc_i in self._bcs_v.items():
                for bc in bc_i:
                    if not isinstance(bc.function_arg, ufl.classes.Zero):
                        inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v[(1)])
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v[(1)]
            bcs_zeta = bcs_v

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)

            v_old = Function(full_space_v, name="v_old")
            zeta_old = Function(full_space_v, name="zeta_old")
            delta_v = Function(full_space_v, name="delta_v")
            delta_zeta = Function(full_space_v, name="delta_zeta")

            v_old.assign(self._v)
            zeta_old.assign(self._zeta)

            if self._initial_condition is not None:
                v_0 = self._initial_condition(v_test)
            else:
                v_0 = Function(space_v, name="v_0")

            v_old.sub(0).assign(v_0)
            zeta_old.sub(n_t - 1).assign(Constant(0.0))

            f = self.construct_f(full_space_v, v_test)

            v_d, true_v = self.construct_v_d(full_space_v, v_test)

            M_v = inner(v_trial, v_test) * dx

            if self._CN:
                flattened_space_v_help = tuple(space_v for i in range(n_t - 1))
                mixed_element_v_help = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_v_help])
                full_space_v_help = FunctionSpace(
                    space_v.mesh(), mixed_element_v_help)

            if self._CN:
                rhs_0, rhs_1 = self.non_linear_res_eval(
                    full_space_v_help, v_old, zeta_old, v_0,
                    bcs_v, bcs_zeta)
            else:
                rhs_0, rhs_1 = self.non_linear_res_eval(
                    full_space_v, v_old, zeta_old, v_0,
                    v_d, f, M_v, bcs_v, bcs_zeta)

            if not self._CN:
                rhs = Function(full_space_v * full_space_v, name="rhs")
                for i in range(n_t):
                    rhs.sub(i).assign(rhs_0.sub(i))
                    rhs.sub(n_t + i).assign(rhs_1.sub(i))
            else:
                rhs = Function(
                    full_space_v_help * full_space_v_help, name="rhs")
                for i in range(n_t - 1):
                    rhs.sub(i).assign(rhs_0.sub(i))
                    rhs.sub(n_t - 1 + i).assign(rhs_1.sub(i))

            with rhs.dat.vec_ro as b_v:
                norm_0 = b_v.norm()
            norm_k = norm_0

            k = 0

            while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):  # noqa: E501
                self.linear_solve(P=P, solver_parameters=solver_parameters,
                                  Multigrid=Multigrid,
                                  lambda_v_bounds=lambda_v_bounds,
                                  v_d=rhs_0, f=rhs_1,
                                  print_error=print_error_linear,
                                  create_output=False,
                                  plots=False)

                delta_v.assign(self._v)
                delta_zeta.assign(self._zeta)

                with delta_v.dat.vec_ro as b_v, \
                        v_old.dat.vec as b_0_v:
                    b_0_v.axpy(1.0, b_v)
                if inhomogeneous_bcs_v:
                    for i in range(n_t):
                        for bc in bcs_v_help[(i)]:
                            bc.apply(v_old.sub(i))

                with delta_zeta.dat.vec_ro as b_v, \
                        zeta_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                for i in range(n_t):
                    for bc in bcs_zeta:
                        bc.apply(zeta_old.sub(i))

                self.set_v(v_old)
                self.set_zeta(zeta_old)

                rhs_0, rhs_1 = self.non_linear_res_eval(
                    full_space_v, v_old, zeta_old, v_0,
                    v_d, f, M_v, bcs_v, bcs_zeta)

                if not self._CN:
                    for i in range(n_t):
                        rhs.sub(i).assign(rhs_0.sub(i))
                        rhs.sub(n_t + i).assign(rhs_1.sub(i))
                else:
                    for i in range(n_t - 1):
                        rhs.sub(i).assign(rhs_0.sub(i))
                        rhs.sub(n_t - 1 + i).assign(rhs_1.sub(i))

                with rhs.dat.vec_ro as b_v:
                    norm_k = b_v.norm()

                k += 1
                if k + 1 > max_non_linear_iter:
                    break

            if print_error_non_linear is True:
                if (norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol):  # noqa: E501
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                    print('number of non-linear iterations: ', k)
                else:
                    print('the non-linear iteration did not converge')
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                self.print_error(full_space_v, v_test)

            if create_output is True:
                v_output = File("v.pvd")
                zeta_output = File("zeta.pvd")

                b_help_v = Function(space_v, name="b_help_v")

                for i in range(n_t):
                    t = t_0 + i * tau

                    b_help_v.interpolate(self._v.sub(i))
                    v_output.write(b_help_v, t=t)

                    b_help_v.interpolate(self._zeta.sub(i))
                    zeta_output.write(b_help_v, t=t)

                del b_help_v

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(self._v)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(self._zeta)

            if plots:
                for i in range(n_t):
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(v.sub(i), axes=axes)
                        fig_v.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(zeta.sub(i), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(i), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)

        def incompressible_linear_solve(self, nullspace_p, *, space_p=None,
                                        P=None, solver_parameters=None,
                                        Multigrid=False,
                                        lambda_v_bounds=None,
                                        lambda_p_bounds=None,
                                        v_d=None, f=None,
                                        div_v=None, div_zeta=None,
                                        print_error=True,
                                        create_output=True, plots=False):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)

            if space_p is None:
                if self._space_p is not None:
                    space_p = self._space_p
                else:
                    raise ValueError("Undefined space_p")
            else:
                self.set_space_p(space_p)
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]
            tau = (T_f - t_0) / (n_t - 1.0)

            beta = self._beta

            if self._CN:
                epsilon = None
            else:
                epsilon = 10.0**-3

            inhomogeneous_bcs_v = False
            for (i), bc_i in self._bcs_v.items():
                for bc in bc_i:
                    if not isinstance(bc.function_arg, ufl.classes.Zero):
                        inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v[(1)])
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v[(1)]
            bcs_zeta = bcs_v

            nullspace_v = DirichletBCNullspace(bcs_v)
            nullspace_zeta = DirichletBCNullspace(bcs_zeta)

            full_nullspace_v = ()
            full_nullspace_zeta = ()
            full_nullspace_p = ()
            for i in range(n_t - 1):
                full_nullspace_v = full_nullspace_v + (nullspace_v, )
                full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )
                full_nullspace_p = full_nullspace_p + (nullspace_p, )
            if not self._CN:
                full_nullspace_v = full_nullspace_v + (nullspace_v, )
                full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )
                full_nullspace_p = full_nullspace_p + (nullspace_p, )

            full_nullspace_0 = full_nullspace_v + full_nullspace_zeta
            full_nullspace_1 = full_nullspace_p + full_nullspace_p

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)
            if not self._CN:
                space_0 = FunctionSpace(
                    space_v.mesh(),
                    full_space_v.ufl_element() * full_space_v.ufl_element())
            else:
                flattened_space_v_help = tuple(
                    space_v for i in range(n_t - 1))
                mixed_element_v_help = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_v_help])
                full_space_v_help = FunctionSpace(
                    space_v.mesh(),
                    mixed_element_v_help)
                space_0 = FunctionSpace(
                    space_v.mesh(),
                    full_space_v_help.ufl_element() * full_space_v_help.ufl_element())  # noqa: E501

            if not self._CN:
                flattened_space_p = tuple(space_p for i in range(n_t))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)
            else:
                flattened_space_p = tuple(space_p for i in range(n_t - 1))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)
            space_1 = FunctionSpace(
                space_p.mesh(),
                full_space_p.ufl_element() * full_space_p.ufl_element())

            b_0 = Cofunction(space_0.dual(), name="b_0")
            b_1 = Cofunction(space_1.dual(), name="b_1")

            if self._CN:
                b_0_0 = Cofunction(full_space_v_help.dual(), name="b_0_0")
                b_0_1 = Cofunction(full_space_v_help.dual(), name="b_0_1")
            else:
                b_0_0 = Cofunction(full_space_v.dual(), name="b_0_0")
                b_0_1 = Cofunction(full_space_v.dual(), name="b_0_1")
            b_1_0 = Cofunction(full_space_p.dual(), name="b_1_0")
            b_1_1 = Cofunction(full_space_p.dual(), name="b_1_1")

            if self._initial_condition is not None:
                v_0 = self._initial_condition(v_test)
            else:
                v_0 = Function(space_v, name="v_0")

            if f is None:
                check_f = True
                f = self.construct_f(full_space_v, v_test)
            else:
                check_f = False

            if v_d is None:
                check_v_d = True
                v_d, true_v = self.construct_v_d(full_space_v, v_test)
            else:
                check_v_d = False
                true_v = Function(full_space_v)
                for i in range(n_t):
                    t = t_0 + i * tau
                    v_d_i, true_v_i = self._desired_state(v_test, t)
                    true_v.sub(i).assign(true_v_i)

            v_old = self._v
            v_n_help = Function(space_v)

            M_v = inner(v_trial, v_test) * dx
            B = - inner(div(v_trial), p_test) * dx
            B_T = - inner(p_trial, div(v_test)) * dx

            block_00 = {}
            block_01 = {}
            block_10 = {}
            block_11 = {}

            if P is None:
                block_00_int = {}
                block_01_int = {}
                block_10_int = {}
                block_11_int = {}

                if self._CN:
                    if self._M_p is not None:
                        block_00_p = 0.5 * tau * self._M_p
                    else:
                        block_00_p = 0.5 * tau * self._J_v(p_trial, p_test)
                    if self._M_mu is not None:
                        block_11_p = - 0.5 * (tau / beta) * self._M_mu
                    else:
                        block_11_p = - 0.5 * (tau / beta) * self._J_u(p_trial,
                                                                      p_test)
                else:
                    if self._M_p is not None:
                        block_00_p = tau * self._M_p
                    else:
                        block_00_p = tau * self._J_v(p_trial, p_test)
                    if self._M_mu is not None:
                        block_11_p = - (tau / beta) * self._M_mu
                    else:
                        block_11_p = - (tau / beta) * self._J_u(p_trial,
                                                                p_test)

                K_p = inner(grad(p_trial), grad(p_test)) * dx
                M_p = inner(p_trial, p_test) * dx

            if not self._CN:
                for i in range(2 * n_t):
                    for j in range(2 * n_t):
                        block_11[(i, j)] = None
                        if j == i:
                            block_01[(i, j)] = tau * B_T
                            block_10[(i, j)] = tau * B
                        else:
                            block_01[(i, j)] = None
                            block_10[(i, j)] = None
            else:
                for i in range(2 * n_t - 2):
                    for j in range(2 * n_t - 2):
                        block_11[(i, j)] = None
                        if j == i:
                            block_01[(i, j)] = tau * B_T
                            block_10[(i, j)] = tau * B
                        else:
                            block_01[(i, j)] = None
                            block_10[(i, j)] = None

            if not self._CN:
                for j in range(n_t):
                    block_00[(n_t, n_t + j)] = None

                    if P is None:
                        block_11_int[(0, j)] = None

            for i in range(n_t - 1):
                t = t_0 + i * tau
                v_n_help.assign(v_old.sub(i))

                D_v_i = self.construct_D_v(v_trial, v_test, v_n_help, t)
                D_zeta_i = adjoint(D_v_i)

                if not self._CN:
                    for j in range(n_t):
                        if j == i - 1:
                            block_00[(i, j)] = None
                            block_00[(i, n_t + j)] = None
                            block_00[(n_t + i, j)] = -M_v
                            block_00[(n_t + i + 1, n_t + j)] = None

                            if P is None:
                                block_00_int[(i, j)] = None
                                block_01_int[(i, j)] = None
                                block_10_int[(i, j)] = -M_v
                                block_11_int[(i + 1, j)] = None
                        elif j == i:
                            block_00[(i, j)] = tau * self._M_v
                            block_00[(i, n_t + j)] = tau * D_zeta_i + M_v
                            block_00[(n_t + i, j)] = tau * D_v_i + M_v
                            block_00[(n_t + i + 1, n_t + j)] = None

                            if P is None:
                                block_00_int[(i, j)] = tau * self._M_v
                                block_01_int[(i, j)] = tau * D_zeta_i + M_v
                                block_10_int[(i, j)] = tau * D_v_i + M_v
                                block_11_int[(i + 1, j)] = None
                        elif j == i + 1:
                            block_00[(i, j)] = None
                            block_00[(i, n_t + j)] = -M_v
                            block_00[(n_t + i, j)] = None
                            block_00[(n_t + i + 1, n_t + j)] = - (tau / beta) * self._M_zeta  # noqa: E501

                            if P is None:
                                block_00_int[(i, j)] = None
                                block_01_int[(i, j)] = -M_v
                                block_10_int[(i, j)] = None
                                block_11_int[(i + 1, j)] = - (tau / beta) * self._M_zeta  # noqa: E501
                        else:
                            block_00[(i, j)] = None
                            block_00[(i, n_t + j)] = None
                            block_00[(n_t + i, j)] = None
                            block_00[(n_t + i + 1, n_t + j)] = None

                            if P is None:
                                block_00_int[(i, j)] = None
                                block_01_int[(i, j)] = None
                                block_10_int[(i, j)] = None
                                block_11_int[(i + 1, j)] = None
                else:
                    v_n_help_plus = Function(space_v)
                    v_n_help_plus.assign(v_old.sub(i + 1))

                    D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                    v_n_help_plus, t + tau)
                    D_zeta_i_plus = adjoint(D_v_i_plus)

                    for j in range(n_t - 1):
                        if j == i - 1:
                            block_00[(i, j)] = 0.5 * tau * self._M_v
                            block_00[(i, n_t + j - 1)] = None
                            block_00[(n_t + i - 1, j)] = 0.5 * tau * D_v_i - M_v  # noqa: E501
                            block_00[(n_t + i - 1, n_t + j - 1)] = None

                            if P is None:
                                block_00_int[(i, j)] = 0.5 * tau * self._M_v
                                block_01_int[(i, j)] = None
                                block_10_int[(i, j)] = 0.5 * tau * D_v_i - M_v
                                block_11_int[(i, j)] = None
                        elif j == i:
                            block_00[(i, j)] = 0.5 * tau * self._M_v
                            block_00[(i, n_t + j - 1)] = 0.5 * tau * D_zeta_i_plus + M_v  # noqa: E501
                            block_00[(n_t + i - 1, j)] = 0.5 * tau * D_v_i_plus + M_v  # noqa: E501
                            block_00[(n_t + i - 1, n_t + j - 1)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501

                            if P is None:
                                block_00_int[(i, j)] = 0.5 * tau * self._M_v
                                block_01_int[(i, j)] = 0.5 * tau * D_zeta_i_plus + M_v  # noqa: E501
                                block_10_int[(i, j)] = 0.5 * tau * D_v_i_plus + M_v  # noqa: E501
                                block_11_int[(i, j)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501
                        elif j == i + 1:
                            block_00[(i, j)] = None
                            block_00[(i, n_t + j - 1)] = 0.5 * tau * D_zeta_i - M_v  # noqa: E501
                            block_00[(n_t + i - 1, j)] = None
                            block_00[(n_t + i - 1, n_t + j - 1)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501

                            if P is None:
                                block_00_int[(i, j)] = None
                                block_01_int[(i, j)] = 0.5 * tau * D_zeta_i - M_v  # noqa: E501
                                block_10_int[(i, j)] = None
                                block_11_int[(i, j)] = - 0.5 * (tau / beta) * self._M_zeta  # noqa: E501
                        else:
                            block_00[(i, j)] = None
                            block_00[(i, n_t + j - 1)] = None
                            block_00[(n_t + i - 1, j)] = None
                            block_00[(n_t + i - 1, n_t + j - 1)] = None

                            if P is None:
                                block_00_int[(i, j)] = None
                                block_01_int[(i, j)] = None
                                block_10_int[(i, j)] = None
                                block_11_int[(i, j)] = None

            if not self._CN:
                t = T_f
                v_n_help.assign(v_old.sub(n_t - 1))

                D_v_i = self.construct_D_v(v_trial, v_test, v_n_help, t)
                D_zeta_i = adjoint(D_v_i)

                for j in range(n_t - 2):
                    block_00[(n_t - 1, j)] = None
                    block_00[(n_t - 1, n_t + j)] = None
                    block_00[(2 * n_t - 1, j)] = None

                    if P is None:
                        block_00_int[(n_t - 1, j)] = None
                        block_01_int[(n_t - 1, j)] = None
                        block_10_int[(n_t - 1, j)] = None

                block_00[(n_t - 1, n_t - 2)] = None
                block_00[(n_t - 1, n_t - 1)] = None
                block_00[(n_t - 1, 2 * n_t - 2)] = None
                block_00[(n_t - 1, 2 * n_t - 1)] = tau * D_zeta_i + M_v
                block_00[(2 * n_t - 1, n_t - 2)] = - M_v
                block_00[(2 * n_t - 1, n_t - 1)] = tau * D_v_i + M_v

                if P is None:
                    block_00_int[(n_t - 1, n_t - 2)] = None
                    block_00_int[(n_t - 1, n_t - 1)] = None
                    block_01_int[(n_t - 1, n_t - 2)] = None
                    block_01_int[(n_t - 1, n_t - 1)] = tau * D_zeta_i + M_v
                    block_10_int[(n_t - 1, n_t - 2)] = - M_v
                    block_10_int[(n_t - 1, n_t - 1)] = tau * D_v_i + M_v

            if not self._CN:
                if check_v_d:
                    b_0_0.sub(0).assign(tau * v_d.sub(0))
                    if inhomogeneous_bcs_v:
                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(0)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * self._M_v, v_inhom))
                        with b_0_0.sub(0).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del v_inhom
                        del b_help
                    for bc in bcs_zeta:
                        bc.apply(b_0_0.sub(0))
                else:
                    b_0_0.sub(0).assign(v_d.sub(0))

                if check_f:
                    D_v_i = self.construct_D_v(v_trial, v_test, v_0, t_0)
                    b_0_1.sub(0).assign(assemble(
                        action(tau * D_v_i + M_v, v_0)))
                    if inhomogeneous_bcs_v:
                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(0)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * D_v_i + M_v, v_inhom))
                        with b_0_1.sub(0).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del v_inhom
                        del b_help
                    for bc in bcs_v:
                        bc.apply(b_0_1.sub(0))
                else:
                    b_0_1.sub(0).assign(f.sub(0))

                for i in range(1, n_t - 1):
                    if check_v_d:
                        b_0_0.sub(i).assign(tau * v_d.sub(i))
                        if inhomogeneous_bcs_v:
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * self._M_v, v_inhom))
                            with b_0_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                        for bc in bcs_zeta:
                            bc.apply(b_0_0.sub(i))
                    else:
                        b_0_0.sub(i).assign(v_d.sub(i))

                    if check_f:
                        b_0_1.sub(i).assign(tau * f.sub(i))
                        if inhomogeneous_bcs_v:
                            t = t_0 + i * tau

                            v_n_help.assign(v_old.sub(i))
                            D_v_i = self.construct_D_v(v_trial, v_test,
                                                       v_n_help, t)

                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * D_v_i + M_v,
                                                     v_inhom))
                            with b_0_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i - 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(M_v, v_inhom))
                            with b_0_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(1.0, b_1_v)
                            del v_inhom
                            del b_help
                        for bc in bcs_v:
                            bc.apply(b_0_1.sub(i))
                    else:
                        b_0_1.sub(i).assign(f.sub(i))

                if not check_v_d:
                    b_0_0.sub(n_t - 1).assign(v_d.sub(n_t - 1))

                if check_f:
                    b_0_1.sub(n_t - 1).assign(tau * f.sub(n_t - 1))
                    if inhomogeneous_bcs_v:
                        v_n_help.assign(v_old.sub(n_t - 1))

                        D_v_i = self.construct_D_v(v_trial, v_test,
                                                   v_n_help, T_f)

                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(n_t - 1)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(tau * D_v_i + M_v, v_inhom))
                        with b_0_1.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)
                        del v_inhom
                        del b_help
                        v_inhom = Function(space_v)
                        for bc in bcs_v_help[(n_t - 2)]:
                            bc.apply(v_inhom)
                        b_help = assemble(action(M_v, v_inhom))
                        with b_0_1.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(1.0, b_1_v)
                        del v_inhom
                        del b_help
                    for bc in bcs_v:
                        bc.apply(b_0_1.sub(n_t - 1))
                else:
                    b_0_1.sub(n_t - 1).assign(f.sub(n_t - 1))

                if div_v is None:
                    if inhomogeneous_bcs_v:
                        for i in range(n_t):
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * B, v_inhom))
                            with b_1_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                else:
                    for i in range(n_t):
                        b_1_0.sub(i).assign(div_v.sub(i))

                if div_zeta is not None:
                    for i in range(n_t):
                        b_1_1.sub(i).assign(div_zeta.sub(i))

                for i in range(n_t):
                    b_0.sub(i).assign(b_0_0.sub(i))
                    index = n_t + i
                    b_0.sub(index).assign(b_0_1.sub(i))
                    b_1.sub(i).assign(b_1_0.sub(i))
                    b_1.sub(index).assign(b_1_1.sub(i))
            else:
                for i in range(n_t - 1):
                    if check_v_d:
                        b_0_0.sub(i).assign(
                            0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1)))
                        if inhomogeneous_bcs_v:
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i + 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(0.5 * tau * self._M_v,
                                                     v_inhom))
                            with b_0_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                            if i > 0:
                                v_inhom = Function(space_v)
                                for bc in bcs_v_help[(i)]:
                                    bc.apply(v_inhom)
                                b_help = assemble(
                                    action(0.5 * tau * self._M_v, v_inhom))
                                with b_0_0.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(-1.0, b_1_v)
                                del v_inhom
                                del b_help
                        for bc in bcs_zeta:
                            bc.apply(b_0_0.sub(i))
                    else:
                        b_0_0.sub(i).assign(v_d.sub(i))

                    if check_f:
                        b_0_1.sub(i).assign(
                            0.5 * tau * (f.sub(i) + f.sub(i + 1)))
                        if inhomogeneous_bcs_v:
                            t = t_0 + (i + 1) * tau

                            v_n_help.assign(v_old.sub(i + 1))

                            D_v_i = self.construct_D_v(v_trial, v_test,
                                                       v_n_help, t)

                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i + 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(0.5 * tau * D_v_i + M_v,
                                                     v_inhom))
                            with b_0_1.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                            if i > 0:
                                t = t_0 + i * tau

                                v_n_help.assign(v_old.sub(i))

                                D_v_i = self.construct_D_v(v_trial, v_test,
                                                           v_n_help, t)

                                v_inhom = Function(space_v)
                                for bc in bcs_v_help[(i)]:
                                    bc.apply(v_inhom)
                                b_help = assemble(
                                    action(0.5 * tau * D_v_i - M_v, v_inhom))
                                with b_0_1.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(-1.0, b_1_v)
                                del v_inhom
                                del b_help
                        for bc in bcs_v:
                            bc.apply(b_0_1.sub(i))
                    else:
                        b_0_1.sub(i).assign(f.sub(i))

                if check_v_d:
                    b = assemble(action(0.5 * tau * self._M_v, v_0))
                    with b_0_0.sub(0).dat.vec as b_v, \
                            b.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_zeta:
                        bc.apply(b_0_0.sub(0))
                    del b

                if check_f:
                    D_v_i = self.construct_D_v(v_trial, v_test, v_0, t_0)

                    b = assemble(action(0.5 * tau * D_v_i - M_v, v_0))
                    with b_0_1.sub(0).dat.vec as b_v, \
                            b.dat.vec_ro as b_1_v:
                        b_v.axpy(-1.0, b_1_v)
                    for bc in bcs_v:
                        bc.apply(b_0_1.sub(0))
                    del b

                if div_v is None:
                    if inhomogeneous_bcs_v:
                        for i in range(n_t - 1):
                            v_inhom = Function(space_v)
                            for bc in bcs_v_help[(i + 1)]:
                                bc.apply(v_inhom)
                            b_help = assemble(action(tau * B, v_inhom))
                            with b_1_0.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            del v_inhom
                            del b_help
                else:
                    for i in range(n_t - 1):
                        b_1_0.sub(i).assign(div_v.sub(i))

                if div_zeta is not None:
                    for i in range(n_t - 1):
                        b_1_1.sub(i).assign(div_zeta.sub(i))

                b_0_0 = apply_T_1(b_0_0, space_v, n_t - 1)
                b_0_1 = apply_T_2(b_0_1, space_v, n_t - 1)

                b_1_0 = apply_T_2(b_1_0, space_p, n_t - 1)
                b_1_1 = apply_T_1(b_1_1, space_p, n_t - 1)

                for i in range(n_t - 1):
                    b_0.sub(i).assign(b_0_0.sub(i))
                    index = n_t - 1 + i
                    b_0.sub(index).assign(b_0_1.sub(i))
                    b_1.sub(i).assign(b_1_0.sub(i))
                    index = n_t - 1 + i
                    b_1.sub(index).assign(b_1_1.sub(i))

            del b_0_0
            del b_0_1
            del b_1_0
            del b_1_1

            if not self._CN:
                system = MultiBlockSystem(
                    space_v, space_p,
                    block_00=block_00, block_01=block_01,
                    block_10=block_10, block_11=block_11,
                    n_blocks_00=2 * n_t, n_blocks_11=2 * n_t,
                    nullspace_0=full_nullspace_0, nullspace_1=full_nullspace_1,
                    CN=self._CN)
            else:
                system = MultiBlockSystem(
                    space_v, space_p,
                    block_00=block_00, block_01=block_01,
                    block_10=block_10, block_11=block_11,
                    n_blocks_00=2 * n_t - 2, n_blocks_11=2 * n_t - 2,
                    sub_n_blocks_00_0=n_t - 1, sub_n_blocks_11_0=n_t - 1,
                    nullspace_0=full_nullspace_0, nullspace_1=full_nullspace_1,
                    CN=self._CN)

            if solver_parameters is None:
                solver_parameters = {"linear_solver": "fgmres",
                                     "fgmres_restart": 10,
                                     "maximum_iterations": 100,
                                     "relative_tolerance": 1.0e-6,
                                     "absolute_tolerance": 0.0,
                                     "monitor_convergence": print_error}

            if P is None:
                solver_K_p = LinearSolver(
                    assemble(K_p),
                    solver_parameters={
                        "ksp_type": "preonly",
                        "pc_type": "hypre",
                        "pc_hypre_type": "boomeramg",
                        "ksp_max_it": 1,
                        "pc_hypre_boomeramg_max_iter": 1,
                        "ksp_atol": 0.0,
                        "ksp_rtol": 0.0})

                if lambda_p_bounds is not None:
                    e_min_p = lambda_p_bounds[0]
                    e_max_p = lambda_p_bounds[1]
                    solver_M_p = LinearSolver(
                        assemble(M_p),
                        solver_parameters={
                            "ksp_type": "chebyshev",
                            "pc_type": "jacobi",
                            "ksp_chebyshev_eigenvalues": f"{e_min_p:.16e}, {e_max_p:.16e}",  # noqa: E501
                            "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                            "ksp_chebyshev_esteig_steps": 0,
                            "ksp_chebyshev_esteig_noisy": False,
                            "ksp_max_it": 20,
                            "ksp_atol": 0.0,
                            "ksp_rtol": 0.0})
                else:
                    solver_M_p = LinearSolver(
                        assemble(M_p),
                        solver_parameters={"ksp_type": "preonly",
                                           "pc_type": "jacobi",
                                           "ksp_max_it": 1,
                                           "ksp_atol": 0.0,
                                           "ksp_rtol": 0.0})

                # definition of preconditioner
                if self._CN:
                    def pc_fn(u_0, u_1, b_0, b_1):
                        b_0_help = Cofunction(full_space_v_help.dual())
                        b_1_help = Cofunction(full_space_v_help.dual())

                        for i in range(n_t - 1):
                            b_0_help.sub(i).assign(b_0.sub(i))
                            index = n_t - 1 + i
                            b_1_help.sub(i).assign(b_0.sub(index))

                        inner_system = MultiBlockSystem(
                            space_v, space_v,
                            block_00=block_00_int, block_01=block_01_int,
                            block_10=block_10_int, block_11=block_11_int,
                            n_blocks_00=n_t - 1, n_blocks_11=n_t - 1,
                            nullspace_0=full_nullspace_v,
                            nullspace_1=full_nullspace_zeta,
                            CN=self._CN)

                        inner_solver_parameters = {
                            "preconditioner": True,
                            "linear_solver": "gmres",
                            "maximum_iterations": 5,
                            "relative_tolerance": 0.0,
                            "absolute_tolerance": 0.0,
                            "monitor_convergence": False}

                        v_help = Function(full_space_v_help)
                        zeta_help = Function(full_space_v_help)

                        inner_pc_fn = self.construct_pc(
                            Multigrid, lambda_v_bounds, full_space_v_help,
                            bcs_v, bcs_zeta, block_01_int, block_10_int,
                            epsilon)

                        try:
                            inner_ksp_solver = inner_system.solve(
                                v_help, zeta_help, b_0_help, b_1_help,
                                solver_parameters=inner_solver_parameters,
                                pc_fn=inner_pc_fn)
                        except ConvergenceError:
                            assert inner_ksp_solver.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                        del b_0_help
                        del b_1_help

                        for i in range(n_t - 1):
                            u_0.sub(i).assign(v_help.sub(i))
                            index = n_t - 1 + i
                            u_0.sub(index).assign(zeta_help.sub(i))

                        del v_help
                        del zeta_help

                        # u_1 = - b_1 + block_10 * u_0
                        b_0_help = Cofunction(full_space_p.dual())
                        b_1_help = Cofunction(full_space_p.dual())
                        v_help = Function(space_v)
                        for i in range(n_t - 1):
                            v_help.assign(u_0.sub(i))
                            b_0_help.sub(i).assign(assemble(action(B, v_help)))
                            with b_0_help.sub(i).dat.vec as b_v:
                                b_v.scale(tau)

                            v_help.assign(u_0.sub(n_t - 1 + i))
                            b_1_help.sub(i).assign(assemble(action(B, v_help)))
                            with b_1_help.sub(i).dat.vec as b_v:
                                b_v.scale(tau)
                        del v_help

                        b_0_help = apply_T_2(b_0_help, space_p, n_t - 1)
                        b_1_help = apply_T_1(b_1_help, space_p, n_t - 1)

                        for i in range(n_t - 1):
                            with b_0_help.sub(i).dat.vec as b_v, \
                                    b_1.sub(i).dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)
                            index = n_t - 1 + i
                            with b_1_help.sub(i).dat.vec as b_v, \
                                    b_1.sub(index).dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                        # solving for the Schur complement approximation
                        for i in range(n_t - 1):
                            with b_0_help.sub(i).dat.vec as b_v:
                                b_v.scale(1.0 / (tau**2))
                            with b_1_help.sub(i).dat.vec as b_v:
                                b_v.scale(1.0 / (tau**2))

                        b_0_help = apply_T_2_inv(b_0_help, space_p, n_t - 1)
                        b_1_help = apply_T_1_inv(b_1_help, space_p, n_t - 1)

                        p_help = Cofunction(space_p.dual())
                        for i in range(n_t - 1):
                            p_help.assign(b_0_help.sub(i))
                            try:
                                solver_K_p.solve(u_1.sub(i),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                            index = n_t - 1 + i
                            p_help.assign(b_1_help.sub(i))
                            try:
                                solver_K_p.solve(u_1.sub(index),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del p_help
                        del b_0_help
                        del b_1_help

                        b_0_help = Cofunction(full_space_p.dual())
                        b_1_help = Cofunction(full_space_p.dual())

                        for i in range(n_t - 1):
                            v_n_help = Function(space_v)
                            v_n_help_plus = Function(space_v)

                            p_help = Function(space_p)
                            mu_help = Function(space_p)

                            t = t_0 + i * tau

                            v_n_help.assign(v_old.sub(i))
                            D_p_i = self.construct_D_v(p_trial, p_test,
                                                       v_n_help, t)
                            D_mu_i = adjoint(D_p_i)

                            v_n_help_plus.assign(v_old.sub(i + 1))
                            D_p_i_plus = self.construct_D_v(
                                p_trial, p_test, v_n_help_plus, t + tau)
                            D_mu_i_plus = adjoint(D_p_i_plus)

                            p_help.assign(u_1.sub(i))
                            b_help = assemble(action(block_00_p, p_help))
                            b_0_help.sub(i).assign(b_help)

                            b_help = assemble(
                                action(0.5 * tau * D_p_i_plus + M_p, p_help))
                            b_1_help.sub(i).assign(b_help)

                            if i > 0:
                                p_help.assign(u_1.sub(i - 1))
                                b_help = assemble(action(block_00_p, p_help))
                                with b_0_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(1.0, b_1_v)

                                b_help = assemble(
                                    action(0.5 * tau * D_p_i - M_p, p_help))
                                with b_1_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(1.0, b_1_v)

                            mu_help.assign(u_1.sub(n_t - 1 + i))
                            b_help = assemble(action(0.5 * tau * D_mu_i + M_p,
                                                     mu_help))
                            with b_0_help.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(1.0, b_1_v)

                            b_help = assemble(action(block_11_p, mu_help))
                            with b_1_help.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(1.0, b_1_v)

                            if i < n_t - 2:
                                mu_help.assign(u_1.sub(n_t + i))
                                b_help = assemble(
                                    action(0.5 * tau * D_mu_i_plus - M_p,
                                           mu_help))
                                with b_0_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(1.0, b_1_v)

                                b_help = assemble(action(block_11_p, mu_help))
                                with b_1_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(1.0, b_1_v)

                        del p_help
                        del mu_help

                        p_help = Cofunction(space_p.dual())
                        for i in range(n_t - 1):
                            p_help.assign(b_0_help.sub(i))
                            try:
                                solver_M_p.solve(u_1.sub(i),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                            index = n_t - 1 + i
                            p_help.assign(b_1_help.sub(i))
                            try:
                                solver_M_p.solve(u_1.sub(index),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del p_help
                        del b_0_help
                        del b_1_help
                else:
                    def pc_fn(u_0, u_1, b_0, b_1):
                        b_0_help = Cofunction(full_space_v.dual())
                        b_1_help = Cofunction(full_space_v.dual())

                        for i in range(n_t):
                            b_0_help.sub(i).assign(b_0.sub(i))
                            index = n_t + i
                            b_1_help.sub(i).assign(b_0.sub(index))

                        inner_system = MultiBlockSystem(
                            space_v, space_v,
                            block_00=block_00_int, block_01=block_01_int,
                            block_10=block_10_int, block_11=block_11_int,
                            n_blocks_00=n_t, n_blocks_11=n_t,
                            nullspace_0=full_nullspace_v,
                            nullspace_1=full_nullspace_zeta,
                            CN=self._CN)

                        inner_solver_parameters = {
                            "preconditioner": True,
                            "linear_solver": "gmres",
                            "maximum_iterations": 5,
                            "relative_tolerance": 0.0,
                            "absolute_tolerance": 0.0,
                            "monitor_convergence": False}

                        v_help = Function(full_space_v)
                        zeta_help = Function(full_space_v)

                        inner_pc_fn = self.construct_pc(
                            Multigrid, lambda_v_bounds, full_space_v,
                            bcs_v, bcs_zeta, block_01_int, block_10_int,
                            epsilon)

                        try:
                            inner_ksp_solver = inner_system.solve(
                                v_help, zeta_help, b_0_help, b_1_help,
                                solver_parameters=inner_solver_parameters,
                                pc_fn=inner_pc_fn)
                        except ConvergenceError:
                            assert inner_ksp_solver.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

                        del b_0_help
                        del b_1_help

                        for i in range(n_t):
                            u_0.sub(i).assign(v_help.sub(i))
                            index = n_t + i
                            u_0.sub(index).assign(zeta_help.sub(i))

                        del v_help
                        del zeta_help

                        # u_1 = - b_1 + block_10 * u_0
                        b_0_help = Cofunction(full_space_p.dual())
                        b_1_help = Cofunction(full_space_p.dual())
                        v_help = Function(space_v)
                        for i in range(n_t):
                            v_help.assign(u_0.sub(i))
                            b_0_help.sub(i).assign(assemble(action(B, v_help)))
                            with b_0_help.sub(i).dat.vec as b_v:
                                b_v.scale(tau)

                            v_help.assign(u_0.sub(n_t + i))
                            b_1_help.sub(i).assign(assemble(action(B, v_help)))
                            with b_1_help.sub(i).dat.vec as b_v:
                                b_v.scale(tau)
                        del v_help

                        for i in range(n_t):
                            with b_0_help.sub(i).dat.vec as b_v, \
                                    b_1.sub(i).dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                            index = n_t + i
                            with b_1_help.sub(i).dat.vec as b_v, \
                                    b_1.sub(index).dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                        # solving for the Schur complement approximation
                        for i in range(n_t):
                            with b_0_help.sub(i).dat.vec as b_v:
                                b_v.scale(1.0 / (tau**2))
                            with b_1_help.sub(i).dat.vec as b_v:
                                b_v.scale(1.0 / (tau**2))

                        p_help = Cofunction(space_p.dual())
                        for i in range(n_t):
                            p_help.assign(b_0_help.sub(i))
                            try:
                                solver_K_p.solve(u_1.sub(i),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                            index = n_t + i
                            p_help.assign(b_1_help.sub(i))
                            try:
                                solver_K_p.solve(u_1.sub(index),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_K_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del p_help
                        del b_0_help
                        del b_1_help

                        b_0_help = Cofunction(full_space_p.dual())
                        b_1_help = Cofunction(full_space_p.dual())

                        for i in range(n_t - 1):
                            v_n_help = Function(space_v)

                            p_help = Function(space_p)
                            mu_help = Function(space_p)

                            t = t_0 + i * tau

                            v_n_help.assign(v_old.sub(i))
                            D_p_i = self.construct_D_v(p_trial, p_test,
                                                       v_n_help, t)
                            D_mu_i = adjoint(D_p_i)

                            p_help.assign(u_1.sub(i))
                            mu_help.assign(u_1.sub(n_t + i))
                            b_help = assemble(action(block_00_p, p_help))
                            b_0_help.sub(i).assign(b_help)

                            b_help = assemble(action(tau * D_p_i + M_p,
                                                     p_help))
                            b_1_help.sub(i).assign(b_help)

                            if i > 0:
                                p_help.assign(u_1.sub(i - 1))
                                b_help = assemble(action(M_p, p_help))
                                with b_1_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(-1.0, b_1_v)

                                b_help = assemble(action(block_11_p, mu_help))
                                with b_1_help.sub(i).dat.vec as b_v, \
                                        b_help.dat.vec_ro as b_1_v:
                                    b_v.axpy(1.0, b_1_v)

                            b_help = assemble(action(tau * D_mu_i + M_p,
                                                     mu_help))
                            with b_0_help.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(1.0, b_1_v)

                            mu_help.assign(u_1.sub(n_t + i + 1))
                            b_help = assemble(action(M_p, mu_help))
                            with b_0_help.sub(i).dat.vec as b_v, \
                                    b_help.dat.vec_ro as b_1_v:
                                b_v.axpy(-1.0, b_1_v)

                        v_n_help.assign(v_old.sub(n_t - 1))
                        D_p_i = self.construct_D_v(p_trial, p_test,
                                                   v_n_help, T_f)
                        D_mu_i = adjoint(D_p_i)

                        p_help.assign(u_1.sub(n_t - 1))
                        b_help = assemble(action(tau * D_p_i + M_p, p_help))
                        b_1_help.sub(n_t - 1).assign(b_help)

                        p_help.assign(u_1.sub(n_t - 2))
                        b_help = assemble(action(M_p, p_help))
                        with b_1_help.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(-1.0, b_1_v)

                        mu_help.assign(u_1.sub(2 * n_t - 1))
                        b_help = assemble(action(block_11_p, mu_help))
                        with b_1_help.sub(n_t - 1).dat.vec as b_v, \
                                b_help.dat.vec_ro as b_1_v:
                            b_v.axpy(1.0, b_1_v)

                        b_help = assemble(action(tau * D_mu_i + M_p, mu_help))
                        b_0_help.sub(n_t - 1).assign(b_help)

                        del b_help
                        del p_help
                        del mu_help

                        p_help = Cofunction(space_p.dual())
                        for i in range(n_t):
                            p_help.assign(b_0_help.sub(i))
                            try:
                                solver_M_p.solve(u_1.sub(i),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                            index = n_t + i
                            p_help.assign(b_1_help.sub(i))
                            try:
                                solver_M_p.solve(u_1.sub(index),
                                                 p_help.copy(deepcopy=True))
                            except ConvergenceError:
                                assert solver_M_p.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
                        del p_help
                        del b_0_help
                        del b_1_help
            else:
                pc_fn = P

            u_0_sol = Function(space_0)
            u_1_sol = Function(space_1)

            system.solve(
                u_0_sol, u_1_sol, b_0, b_1,
                solver_parameters=solver_parameters,
                pc_fn=pc_fn)

            v = Function(full_space_v, name="v")
            zeta = Function(full_space_v, name="zeta")

            p = Function(full_space_p, name="p")
            mu = Function(full_space_p, name="mu")

            if self._CN:
                if check_f and check_v_d:
                    v.sub(0).assign(v_0)

                for i in range(n_t - 1):
                    index = n_t - 1 + i

                    v.sub(i + 1).assign(u_0_sol.sub(i))
                    p.sub(i).assign(u_1_sol.sub(index))

                    zeta.sub(i).assign(u_0_sol.sub(index))
                    mu.sub(i).assign(u_1_sol.sub(i))
            else:
                for i in range(n_t):
                    index = n_t + i

                    v.sub(i).assign(u_0_sol.sub(i))
                    p.sub(i).assign(u_1_sol.sub(index))

                    zeta.sub(i).assign(u_0_sol.sub(index))
                    mu.sub(i).assign(u_1_sol.sub(i))

                if check_f and check_v_d:
                    v.sub(0).assign(v_0)

            self.set_v(v)
            self.set_zeta(zeta)

            self.set_p(p)
            self.set_mu(mu)

            if print_error:
                self.print_error(full_space_v, v_test)

            if create_output:
                v_output = File("v.pvd")
                zeta_output = File("zeta.pvd")

                p_output = File("p.pvd")
                mu_output = File("mu.pvd")

                b_help_v = Function(space_v, name="b_help_v")
                b_help_p = Function(space_p, name="b_help_p")

                for i in range(n_t - 1):
                    t = t_0 + i * tau

                    b_help_v.interpolate(v.sub(i))
                    v_output.write(b_help_v, t=t)

                    b_help_v.interpolate(zeta.sub(i))
                    zeta_output.write(b_help_v, t=t)

                    if self._CN:
                        t_help = t + 0.5 * tau

                        b_help_p.interpolate(p.sub(i))
                        p_output.write(b_help_p, t=t_help)

                        b_help_p.interpolate(mu.sub(i))
                        mu_output.write(b_help_p, t=t_help)
                    else:
                        b_help_p.interpolate(p.sub(i))
                        p_output.write(b_help_p, t=t)

                        b_help_p.interpolate(mu.sub(i))
                        mu_output.write(b_help_p, t=t)

                t = T_f

                b_help_v.interpolate(v.sub(n_t - 1))
                v_output.write(b_help_v, t=t)

                b_help_v.interpolate(zeta.sub(n_t - 1))
                zeta_output.write(b_help_v, t=t)

                if not self._CN:
                    b_help_p.interpolate(p.sub(n_t - 1))
                    p_output.write(b_help_p, t=t)

                    b_help_p.interpolate(mu.sub(n_t - 1))
                    mu_output.write(b_help_p, t=t)

                del b_help_v
                del b_help_p

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(v)

                with CheckpointFile("p.h5", "w") as h:
                    h.save_function(p)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(zeta)

                with CheckpointFile("mu.h5", "w") as h:
                    h.save_function(mu)

            if plots:
                for i in range(n_t - 1):
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(v.sub(i), axes=axes)
                        fig_v.colorbar(colors)
                        fig_p, axes = plt.subplots()
                        colors = tripcolor(p.sub(i), axes=axes)
                        fig_p.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(zeta.sub(i), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_mu, axes = plt.subplots()
                        colors = tripcolor(mu.sub(i), axes=axes)
                        fig_mu.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(i), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)

                if self._CN:
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(v.sub(n_t - 1), axes=axes)
                        fig_v.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(zeta.sub(n_t - 1), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(n_t - 1), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)
                else:
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(v.sub(n_t - 1), axes=axes)
                        fig_v.colorbar(colors)
                        fig_p, axes = plt.subplots()
                        colors = tripcolor(p.sub(n_t - 1), axes=axes)
                        fig_p.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(zeta.sub(n_t - 1), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_mu, axes = plt.subplots()
                        colors = tripcolor(mu.sub(n_t - 1), axes=axes)
                        fig_mu.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(n_t - 1), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)

        def incompressible_non_linear_solve(self, nullspace_p, *,
                                            space_p=None, P=None,
                                            solver_parameters=None,
                                            Multigrid=False,
                                            lambda_v_bounds=None,
                                            lambda_p_bounds=None,
                                            max_non_linear_iter=10,
                                            relative_non_linear_tol=10.0**-5,
                                            absolute_non_linear_tol=10.0**-8,
                                            print_error_linear=False,
                                            print_error_non_linear=True,
                                            create_output=True,
                                            plots=False):
            space_v = self._space_v
            v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
            if space_p is None:
                if self._space_p is not None:
                    space_p = self._space_p
                else:
                    raise ValueError("Undefined space_p")
            else:
                self.set_space_p(space_p)
            p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

            n_t = self._n_t
            t_0 = self._time_interval[0]
            T_f = self._time_interval[1]
            tau = (T_f - t_0) / (n_t - 1.0)

            inhomogeneous_bcs_v = False
            for (i), bc_i in self._bcs_v.items():
                for bc in bc_i:
                    if not isinstance(bc.function_arg, ufl.classes.Zero):
                        inhomogeneous_bcs_v = True

            if inhomogeneous_bcs_v:
                bcs_v = homogenize(self._bcs_v[(1)])
                bcs_v_help = self._bcs_v
            else:
                bcs_v = self._bcs_v[(1)]
            bcs_zeta = bcs_v

            flattened_space_v = tuple(space_v for i in range(n_t))
            mixed_element_v = ufl.classes.MixedElement(
                *[space.ufl_element() for space in flattened_space_v])
            full_space_v = FunctionSpace(space_v.mesh(), mixed_element_v)
            if self._CN:
                flattened_space_v_help = tuple(
                    space_v for i in range(n_t - 1))
                mixed_element_v_help = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_v_help])
                full_space_v_help = FunctionSpace(
                    space_v.mesh(),
                    mixed_element_v_help)

            if not self._CN:
                flattened_space_p = tuple(space_p for i in range(n_t))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)
            else:
                flattened_space_p = tuple(space_p for i in range(n_t - 1))
                mixed_element_p = ufl.classes.MixedElement(
                    *[space.ufl_element() for space in flattened_space_p])
                full_space_p = FunctionSpace(space_p.mesh(), mixed_element_p)

            v_old = Function(full_space_v, name="v_old")
            zeta_old = Function(full_space_v, name="zeta_old")
            delta_v = Function(full_space_v, name="delta_v")
            delta_zeta = Function(full_space_v, name="delta_zeta")

            p_old = Function(full_space_p, name="p_old")
            mu_old = Function(full_space_p, name="mu_old")
            delta_p = Function(full_space_p, name="delta_p")
            delta_mu = Function(full_space_p, name="delta_mu")

            v_old.assign(self._v)
            zeta_old.assign(self._zeta)

            p_old.assign(self._p)
            mu_old.assign(self._mu)

            if self._initial_condition is not None:
                v_0 = self._initial_condition(v_test)
            else:
                v_0 = Function(space_v, name="v_0")

            v_old.sub(0).assign(v_0)

            zeta_old.sub(n_t - 1).assign(Constant(0.0))

            f = self.construct_f(full_space_v, v_test)

            v_d, true_v = self.construct_v_d(full_space_v, v_test)

            M_v = inner(v_trial, v_test) * dx

            B = - inner(div(v_trial), p_test) * dx
            B_T = - inner(p_trial, div(v_test)) * dx

            def non_linear_res_eval():
                rhs_10 = Cofunction(full_space_p.dual(), name="rhs_10")
                rhs_11 = Cofunction(full_space_p.dual(), name="rhs_11")

                if self._CN:
                    rhs_00 = Cofunction(full_space_v_help.dual(),
                                        name="rhs_00")
                    rhs_01 = Cofunction(full_space_v_help.dual(),
                                        name="rhs_01")

                    rhs_0, rhs_1 = self.non_linear_res_eval(
                        full_space_v_help, v_old, zeta_old, v_0,
                        v_d, f, M_v, bcs_v, bcs_zeta)
                else:
                    rhs_00 = Cofunction(full_space_v.dual(), name="rhs_00")
                    rhs_01 = Cofunction(full_space_v.dual(), name="rhs_01")

                    rhs_0, rhs_1 = self.non_linear_res_eval(
                        full_space_v, v_old, zeta_old, v_0,
                        v_d, f, M_v, bcs_v, bcs_zeta)

                rhs_00.assign(rhs_0)
                rhs_01.assign(rhs_1)

                del rhs_0
                del rhs_1

                if not self._CN:
                    b_p_help = Function(space_p)
                    b_p_help.assign(mu_old.sub(0))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_00.sub(0).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(0))
                    b = assemble(action(tau * B, b_help))
                    rhs_11.sub(0).assign(-b)
                    del b
                    del b_help

                    for bc in bcs_zeta:
                        bc.apply(rhs_00.sub(0))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(n_t - 1))
                    b = assemble(action(tau * B, b_help))
                    rhs_10.sub(n_t - 1).assign(-b)
                    del b
                    del b_help

                    b_p_help = Function(space_p)
                    b_p_help.assign(mu_old.sub(n_t - 1))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_00.sub(n_t - 1).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    for bc in bcs_zeta:
                        bc.apply(rhs_00.sub(n_t - 1))

                    b_p_help = Function(space_p)
                    b_p_help.assign(p_old.sub(0))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_01.sub(0).dat.vec as b_1_v:
                        b_1_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    for bc in bcs_v:
                        bc.apply(rhs_01.sub(0))

                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(0))
                    b = assemble(action(tau * B, b_help))
                    rhs_10.sub(0).assign(-b)
                    del b
                    del b_help

                    b_p_help = Function(space_p)
                    b_p_help.assign(p_old.sub(n_t - 1))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_01.sub(n_t - 1).dat.vec as b_1_v:
                        b_1_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    for bc in bcs_v:
                        bc.apply(rhs_01.sub(n_t - 1))

                    for i in range(1, n_t - 1):
                        b_help = Function(space_v)
                        b_help.assign(v_old.sub(i))
                        b = assemble(action(tau * B, b_help))
                        rhs_10.sub(i).assign(-b)
                        del b
                        del b_help

                        b_p_help = Function(space_p)
                        b_p_help.assign(mu_old.sub(i))
                        b = assemble(action(tau * B_T, b_p_help))
                        with b.dat.vec_ro as b_v, \
                                rhs_00.sub(i).dat.vec as b_0_v:
                            b_0_v.axpy(-1.0, b_v)
                        del b
                        del b_p_help

                        for bc in bcs_zeta:
                            bc.apply(rhs_00.sub(i))

                        b_p_help = Function(space_p)
                        b_p_help.assign(p_old.sub(i))
                        b = assemble(action(tau * B_T, b_p_help))
                        with b.dat.vec_ro as b_v, \
                                rhs_01.sub(i).dat.vec as b_0_v:
                            b_0_v.axpy(-1.0, b_v)
                        del b
                        del b_p_help

                        b_help = Function(space_v)
                        b_help.assign(zeta_old.sub(i))
                        b = assemble(action(tau * B, b_help))
                        rhs_11.sub(i).assign(-b)
                        del b
                        del b_help

                        for bc in bcs_v:
                            bc.apply(rhs_01.sub(i))
                else:
                    b_help = Function(space_v)
                    b_help.assign(v_old.sub(1))
                    b = assemble(action(tau * B, b_help))
                    rhs_10.sub(0).assign(-b)
                    del b
                    del b_help

                    b_p_help = Function(space_p)
                    b_p_help.assign(mu_old.sub(0))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_00.sub(0).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    for bc in bcs_zeta:
                        bc.apply(rhs_00.sub(0))

                    b_help = Function(space_v)
                    b_help.assign(zeta_old.sub(0))
                    b = assemble(action(tau * B, b_help))
                    rhs_11.sub(0).assign(-b)
                    del b
                    del b_help

                    b_p_help = Function(space_p)
                    b_p_help.assign(p_old.sub(0))
                    b = assemble(action(tau * B_T, b_p_help))
                    with b.dat.vec_ro as b_v, \
                            rhs_01.sub(0).dat.vec as b_0_v:
                        b_0_v.axpy(-1.0, b_v)
                    del b
                    del b_p_help

                    for bc in bcs_v:
                        bc.apply(rhs_01.sub(0))

                    for i in range(1, n_t - 1):
                        b_help = Function(space_v)
                        b_help.assign(v_old.sub(i + 1))
                        b = assemble(action(tau * B, b_help))
                        rhs_10.sub(i).assign(-b)
                        del b
                        del b_help

                        b_help = Function(space_v)
                        b_help.assign(zeta_old.sub(i))
                        b = assemble(action(tau * B, b_help))
                        rhs_11.sub(i).assign(-b)
                        del b
                        del b_help

                        b_p_help = Function(space_p)
                        b_p_help.assign(mu_old.sub(i))
                        b = assemble(action(tau * B_T, b_p_help))
                        with b.dat.vec_ro as b_v, \
                                rhs_00.sub(i).dat.vec as b_0_v:
                            b_0_v.axpy(-1.0, b_v)
                        del b
                        del b_p_help

                        for bc in bcs_zeta:
                            bc.apply(rhs_00.sub(i))

                        b_p_help = Function(space_p)
                        b_p_help.assign(p_old.sub(i))
                        b = assemble(action(tau * B_T, b_p_help))
                        with b.dat.vec_ro as b_v, \
                                rhs_01.sub(i).dat.vec as b_0_v:
                            b_0_v.axpy(-1.0, b_v)
                        del b
                        del b_p_help

                        for bc in bcs_v:
                            bc.apply(rhs_01.sub(i))

                return rhs_00, rhs_01, rhs_10, rhs_11

            rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

            if not self._CN:
                rhs = Cofunction((full_space_v * full_space_v * full_space_p * full_space_p).dual(), name="rhs")  # noqa: E501
                for i in range(n_t):
                    rhs.sub(i).assign(rhs_00.sub(i))
                    rhs.sub(n_t + i).assign(rhs_01.sub(i))
                    rhs.sub(2 * n_t + i).assign(rhs_10.sub(i))
                    rhs.sub(3 * n_t + i).assign(rhs_11.sub(i))
            else:
                rhs = Cofunction((full_space_v_help * full_space_v_help * full_space_p * full_space_p).dual(), name="rhs")  # noqa: E501
                for i in range(n_t - 1):
                    rhs.sub(i).assign(rhs_00.sub(i))
                    rhs.sub(n_t - 1 + i).assign(rhs_01.sub(i))
                    rhs.sub(2 * n_t - 2 + i).assign(rhs_10.sub(i))
                    rhs.sub(3 * n_t - 3 + i).assign(rhs_11.sub(i))

            with rhs.dat.vec_ro as b_v:
                norm_0 = b_v.norm()
            norm_k = norm_0

            k = 0

            while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):  # noqa: E501
                self.incompressible_linear_solve(
                    nullspace_p, space_p=space_p,
                    P=P, solver_parameters=solver_parameters,
                    Multigrid=Multigrid,
                    lambda_v_bounds=lambda_v_bounds,
                    lambda_p_bounds=lambda_p_bounds,
                    v_d=rhs_00, f=rhs_01,
                    div_v=rhs_10, div_zeta=rhs_11,
                    print_error=print_error_linear,
                    create_output=False,
                    plots=False)

                delta_v.assign(self._v)
                delta_zeta.assign(self._zeta)

                delta_p.assign(self._p)
                delta_mu.assign(self._mu)

                with delta_v.dat.vec_ro as b_v, \
                        v_old.dat.vec as b_0_v:
                    b_0_v.axpy(1.0, b_v)
                if inhomogeneous_bcs_v:
                    for i in range(n_t):
                        for bc in bcs_v_help[(i)]:
                            bc.apply(v_old.sub(i))

                with delta_p.dat.vec_ro as b_v, \
                        p_old.dat.vec as b_0_v:
                    b_0_v.axpy(1.0, b_v)

                with delta_zeta.dat.vec_ro as b_v, \
                        zeta_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)
                for i in range(n_t):
                    for bc in bcs_zeta:
                        bc.apply(zeta_old.sub(i))

                with delta_mu.dat.vec_ro as b_v, \
                        mu_old.dat.vec as b_1_v:
                    b_1_v.axpy(1.0, b_v)

                self.set_v(v_old)
                self.set_zeta(zeta_old)

                self.set_p(p_old)
                self.set_mu(mu_old)

                rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

                if not self._CN:
                    for i in range(n_t):
                        rhs.sub(i).assign(rhs_00.sub(i))
                        rhs.sub(n_t + i).assign(rhs_01.sub(i))
                        rhs.sub(2 * n_t + i).assign(rhs_10.sub(i))
                        rhs.sub(3 * n_t + i).assign(rhs_11.sub(i))
                else:
                    for i in range(n_t - 1):
                        rhs.sub(i).assign(rhs_00.sub(i))
                        rhs.sub(n_t - 1 + i).assign(rhs_01.sub(i))
                        rhs.sub(2 * n_t - 2 + i).assign(rhs_10.sub(i))
                        rhs.sub(3 * n_t - 3 + i).assign(rhs_11.sub(i))

                with rhs.dat.vec_ro as b_v:
                    norm_k = b_v.norm()

                k += 1
                if k + 1 > max_non_linear_iter:
                    break

            if print_error_non_linear:
                if (norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol):  # noqa: E501
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                    print('number of non-linear iterations: ', k)
                else:
                    print('the non-linear iteration did not converge')
                    print('relative non-linear residual: ', norm_k / norm_0)
                    print('absolute non-linear residual: ', norm_k)
                v_err = self._v - true_v
                print('estimated error in the L2-norm: ',
                      sqrt(tau) * sqrt(abs(assemble(inner(v_err,
                                                          v_err) * dx))))

            if create_output:
                v_output = File("v.pvd")
                zeta_output = File("zeta.pvd")

                p_output = File("p.pvd")
                mu_output = File("mu.pvd")

                b_help_v = Function(space_v, name="b_help_v")
                b_help_p = Function(space_p, name="b_help_p")

                for i in range(n_t - 1):
                    t = t_0 + i * tau

                    b_help_v.interpolate(self._v.sub(i))
                    v_output.write(b_help_v, t=t)

                    b_help_v.interpolate(self._zeta.sub(i))
                    zeta_output.write(b_help_v, t=t)

                    if self._CN:
                        t_help = t + 0.5 * tau
                        b_help_p.interpolate(self._p.sub(i))
                        p_output.write(b_help_p, t=t_help)

                        b_help_p.interpolate(self._mu.sub(i))
                        mu_output.write(b_help_p, t=t_help)
                    else:
                        b_help_p.interpolate(self._p.sub(i))
                        p_output.write(b_help_p, t=t)

                        b_help_p.interpolate(self._mu.sub(i))
                        mu_output.write(b_help_p, t=t)

                t = T_f
                b_help_v.interpolate(self._v.sub(n_t - 1))
                v_output.write(b_help_v, t=t)

                b_help_v.interpolate(self._zeta.sub(n_t - 1))
                zeta_output.write(b_help_v, t=t)

                if not self._CN:
                    b_help_p.interpolate(self._p.sub(n_t - 1))
                    p_output.write(b_help_p, t=t)

                    b_help_p.interpolate(self._mu.sub(i))
                    mu_output.write(b_help_p, t=t)

                del b_help_v
                del b_help_p

                with CheckpointFile("v.h5", "w") as h:
                    h.save_function(self._v)

                with CheckpointFile("p.h5", "w") as h:
                    h.save_function(self._p)

                with CheckpointFile("zeta.h5", "w") as h:
                    h.save_function(self._zeta)

                with CheckpointFile("mu.h5", "w") as h:
                    h.save_function(self._mu)

            if plots:
                for i in range(n_t - 1):
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(self._v.sub(i), axes=axes)
                        fig_v.colorbar(colors)
                        fig_p, axes = plt.subplots()
                        colors = tripcolor(self._p.sub(i), axes=axes)
                        fig_p.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(self._zeta.sub(i), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_mu, axes = plt.subplots()
                        colors = tripcolor(self._mu.sub(i), axes=axes)
                        fig_mu.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(i), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)

                if self._CN:
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(self._v.sub(n_t - 1), axes=axes)
                        fig_v.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(self._zeta.sub(n_t - 1), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(n_t - 1), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)
                else:
                    try:
                        import matplotlib.pyplot as plt
                        fig_v, axes = plt.subplots()
                        colors = tripcolor(self._v.sub(n_t - 1), axes=axes)
                        fig_v.colorbar(colors)
                        fig_p, axes = plt.subplots()
                        colors = tripcolor(self._p.sub(n_t - 1), axes=axes)
                        fig_p.colorbar(colors)
                        fig_zeta, axes = plt.subplots()
                        colors = tripcolor(self._zeta.sub(n_t - 1), axes=axes)
                        fig_zeta.colorbar(colors)
                        fig_mu, axes = plt.subplots()
                        colors = tripcolor(self._mu.sub(n_t - 1), axes=axes)
                        fig_mu.colorbar(colors)
                        fig_true_v, axes = plt.subplots()
                        colors = tripcolor(true_v.sub(n_t - 1), axes=axes)
                        fig_true_v.colorbar(colors)
                        plt.show()
                    except Exception as e:
                        warning("Cannot plot figure. Error msg: '%s'" % e)
