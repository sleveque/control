"""A library for solving certain PDE-constrained optimization problems. Uses
Firedrake to derive the finite element discretization of the problems
considered, and the Python interface to PETSc for the derivation of the KKT
conditions and the definition of the linear solvers.
"""

from firedrake import (
    CheckpointFile, Cofunction, Constant, Function, FunctionSpace,
    LinearSolver, MixedFunctionSpace, TestFunction, TrialFunction,
    ZeroBaseForm, action, adjoint, assemble, div, dx, grad, homogenize, inner,
    norm, sqrt, tripcolor
)
from firedrake.functionspaceimpl import WithGeometry as FunctionSpaceBase
from firedrake.output import VTKFile as File

from .cn import apply_T_1, apply_T_2, apply_T_1_inv, apply_T_2_inv
from .preconditioner import DirichletBCNullspace, MultiBlockSystem, apply_bcs

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
import petsc4py.PETSc as PETSc
import ufl

from collections.abc import Sequence
from functools import cached_property, wraps


__all__ = \
    [
        "Stationary",
        "Instationary"
    ]


# Used to avoid convergence errors when max_it is reached
def converged(ksp, it, rnorm):
    return it >= ksp.max_it


def garbage_cleanup(comm):
    def wrapper(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return_value = fn(*args, **kwargs)
            PETSc.garbage_cleanup(comm)
            return return_value
        return wrapped_fn
    return wrapper


def garbage_cleanup_method(attr_name="comm"):
    def wrapper(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            return_value = fn(self, *args, **kwargs)
            PETSc.garbage_cleanup(getattr(self, attr_name))
            return return_value
        return wrapped_fn
    return wrapper


def plot(*u):
    if matplotlib is None:
        raise RuntimeError("matplotlib not available")

    for u_i in u:
        fig, axes = plt.subplots()
        colors = tripcolor(u_i, axes=axes)
        fig.colorbar(colors)
    plt.show()


def output(data):
    for name, u in data.items():
        output = File(f"{name}.pvd")
        output.write(u)
        with CheckpointFile(f"{name}.h5", mode="w") as h:
            h.save_function(u)


def vnorm(u):
    with u.dat.vec_ro as u_v:
        return u_v.norm()


def time(time_interval, i, n_t):
    t_0, t_1 = time_interval
    # Linearly interpolate
    return (t_0 * (n_t - 1 - i) + t_1 * i) / (n_t - 1)


def mass(space):
    return inner(TrialFunction(space), TestFunction(space)) * dx


class Stationary:
    def __init__(self, space_v, forward_form, desired_state=None,
                 force_function=None, *, beta=1.0e-3, space_p=None,
                 Gauss_Newton=False, bcs_v=None):
        """Stationary control problem.

        Input:
            - space_v             space to which the solution belongs

            - forward_form        form that represents the differential
                                  operator in space

            - desired_state       desired state, defaults to zero

            - force_function      force function acting on the system,
                                  defaults to zero

            - beta                regularization parameter

            - space_p             pressure space (only for incompressible
                                  problems)

            - Gauss_Newton        if True, a Gauss--Newton linearization is
                                  employed, otherwise a Picard
                                  linearization is applied

            - bcs_v               boundary conditions on the state
        """

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

        if desired_state is None:
            def desired_state(test_v):
                return ZeroBaseForm((test_v,))
        if force_function is None:
            def force_function(test_v):
                return ZeroBaseForm((test_v,))

        self._space_v = space_v
        self._forward_form = forward_form
        self._desired_state = desired_state
        self._force_function = force_function
        self._beta = beta
        self._Gauss_Newton = Gauss_Newton
        self._bcs_v = bcs_v

        self._v = Function(space_v, name="v")
        apply_bcs(self._bcs_v, self._v)
        self._zeta = Function(space_v, name="zeta")
        self._M_v = self._M_zeta = mass(space_v)

        self._space_p = None
        self._M_p = self._M_mu = None
        if space_p is not None:
            self.set_space_p(space_p)

    @property
    def space_v(self):
        return self._space_v

    @cached_property
    def comm(self):
        return self.space_v.mesh().comm

    @property
    def beta(self):
        return self._beta

    def set_space_p(self, space_p):
        """
        Input:
            - space_p        new pressure space
        """

        self._space_p = space_p
        self._p = Function(space_p, name="p")
        self._mu = Function(space_p, name="mu")
        self._M_p = self._M_mu = mass(space_p)

    def set_v(self, v_new):
        """
        Input:
            - v_new        new approximation of the state solution
        """

        v = Function(self.space_v, name="v").assign(v_new)
        apply_bcs(self._bcs_v, v)
        self._v.assign(v)

    def set_zeta(self, zeta_new):
        """
        Input:
            - zeta_new        new approximation of the adjoint solution
        """

        zeta = Function(self.space_v, name="zeta").assign(zeta_new)
        apply_bcs(homogenize(self._bcs_v), zeta)
        self._zeta.assign(zeta)

    def set_p(self, p_new):
        """
        Input:
            - p_new        new approximation of the pressure state solution
        """

        self._p.assign(p_new)

    def set_mu(self, mu_new):
        """
        Input:
            - mu_new        new approximation of the pressure adjoint
                            solution
        """

        self._mu.assign(mu_new)

    def print_error(self):
        """Print L^2 norm of the difference between the numerical solution
        and the desired state.
        """

        error_norm = norm(self._v - self._true_v, norm_type="L2")
        print(f'Estimated error in the L2-norm: {error_norm:.16e}')

    def construct_D_v(self, v_trial, v_test, v_old, *,
                      non_linear_res=False):
        """Construction of the discretized forward form.

        Input:
            - v_trial               trial function

            - v_test                test function

            - v_old                 approximation of the state solution

            - non_linear_res        if True, the form is employed in the
                                    evaluation of the non-linear residual

        Output:
            - D_v                   discretized forward form
        """

        if not self._Gauss_Newton or non_linear_res:
            # if Gauss--Newton is not applied or we want to
            # evaluate the residual, we take the Picard linearization
            # of the forward form
            return self._forward_form(v_trial, v_test, v_old)
        else:
            # if we want to apply Gauss--Newton, we take the
            # derivative of the form in the direction of v_old
            return ufl.derivative(
                self._forward_form(v_old, v_test, v_old),
                v_old, v_trial)

    def construct_f(self, v_test, D_v, bcs_v, *, v_inhom=None):
        """Construction of the vector containing the force function.

        Input:
            - v_test                     test function

            - D_v                        discretized forward form

            - bcs_v                      homogenization of the bcs on the
                                         state variable

            - v_inhom                    function that is zero in the
                                         interior of the domain and
                                         interpolates the state variable on
                                         the boundary

        Output:
            - f                          discretized force function
        """

        if v_inhom is not None:
            f = assemble(self._force_function(v_test)
                         - action(D_v, v_inhom))
        else:
            f = assemble(self._force_function(v_test))
        apply_bcs(bcs_v, f)
        return f

    def construct_v_d(self, v_test, bcs_v, *, v_inhom=None):
        """Construction of the vector containing the desired state.

        Input:
            - v_test                     test function

            - bcs_v                      homogenization of the bcs on the
                                         state variable

            - v_inhom                    function that is zero in the
                                         interior of the domain and
                                         interpolates the state variable on
                                         the boundary

        Output:
            - v_d                        discretized desired state
        """

        v_d, true_v = self._desired_state(v_test)
        if v_inhom is not None:
            v_d = assemble(v_d - action(self._M_v, v_inhom))
        else:
            v_d = assemble(v_d)
        apply_bcs(bcs_v, v_d)

        self._true_v = true_v
        return v_d

    def construct_pc(self, auxiliary_sp,
                     bcs_v, bcs_zeta, D_v, D_zeta):
        """Construction of the preconditioner, based on the matching strategy.

        Input:
            - auxiliary_sp        auxiliary solver parameters for inner
                                  blocks

            - bcs_v               homogenized boundary conditions for the
                                  state variable

            - bcs_zeta            homogenized boundary conditions for the
                                  adjoint variable

            - D_v                 discretized forward form

            - D_zeta              discretized adjoint form

        Output:
            - pc_linear           preconditioner to employ within Krylov
                                  method
        """

        if "sp_11block" in auxiliary_sp:
            sp_11block = auxiliary_sp["sp_11block"]
        else:
            sp_11block = {"ksp_type": "preonly",
                          "pc_type": "jacobi",
                          "ksp_max_it": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

        if "sp_Schur" in auxiliary_sp:
            sp_Schur = auxiliary_sp["sp_Schur"]
        else:
            sp_Schur = {"ksp_type": "preonly",
                        "pc_type": "hypre",
                        "pc_hypre_type": "boomeramg",
                        "ksp_max_it": 1,
                        "pc_hypre_boomeramg_max_iter": 2,
                        "ksp_atol": 0.0,
                        "ksp_rtol": 0.0}

        solver_0 = LinearSolver(
            assemble(self._M_v, bcs=bcs_v),
            solver_parameters=sp_11block)
        solver_1 = LinearSolver(
            assemble(D_v + (1.0 / self.beta**0.5) * self._M_v,
                     bcs=bcs_zeta),
            solver_parameters=sp_Schur)
        solver_2 = LinearSolver(
            assemble(D_zeta + (1.0 / self.beta**0.5) * self._M_zeta,
                     bcs=bcs_zeta),
            solver_parameters=sp_Schur)
        solver_0.ksp.addConvergenceTest(converged, prepend=True)
        solver_1.ksp.addConvergenceTest(converged, prepend=True)
        solver_2.ksp.addConvergenceTest(converged, prepend=True)

        @garbage_cleanup(self.comm)
        def pc_linear(u_0, u_1, b_0, b_1):
            # solving for the (1,1)-block
            u_0.zero()
            solver_0.solve(u_0, b_0.copy(deepcopy=True))

            # solving for the Schur complement approximation
            # first solve
            # u_1 = - b_1 + D_v * u_0
            b = assemble(action(D_v, u_0) - b_1)
            apply_bcs(bcs_zeta, b)
            u_1.zero()
            solver_1.solve(u_1, b.copy(deepcopy=True))
            # second solve
            b = assemble(action(self._M_v, u_1))
            apply_bcs(bcs_zeta, b)
            u_1.zero()
            solver_2.solve(u_1, b.copy(deepcopy=True))

        return pc_linear

    @garbage_cleanup_method()
    def non_linear_res_eval(self, space_v, v_d, f, v_old, zeta_old,
                            D_v, D_zeta, M_zeta, bcs_v, bcs_zeta):
        """Construction of the non-linear residual.

        Input:
            - space_v      space of state and adjoint variables

            - v_d          desired state

            - f            force function

            - v_old        approximation of state variable

            - zeta_old     approximation of adjoint variable

            - D_v          discretized forward form

            - D_zeta       discretized adjoint form

            - M_zeta       (2,2)-block

            - bcs_v        homogenized boundary conditions for the state
                           variable

            - bcs_zeta     homogenized boundary conditions for the adjoint
                           variable

        Output:
            - rhs_0        non-linear residual (adjoint equation)

            - rhs_1        non-linear residual (state equation)
        """
        # evaluating non-linear residual (adjoint equation)
        rhs_0 = assemble(v_d - action(self._M_v, v_old) - action(D_zeta, zeta_old))
        apply_bcs(bcs_v, rhs_0)

        # evaluating non-linear residual (state equation)
        rhs_1 = assemble(f - action(D_v, v_old) - action(M_zeta, zeta_old))
        apply_bcs(bcs_zeta, rhs_1)

        return rhs_0, rhs_1

    @garbage_cleanup_method()
    def linear_solve(self, *,
                     P=None, solver_parameters=None,
                     auxiliary_sp={}, v_d=None, f=None,
                     print_error=True, create_output=True,
                     plots=False):
        """Module for the solution of linear control problems.

        Input:
            - P                        preconditioner to apply within
                                       the Krylov method (if None, default
                                       option is employed)

            - solver_parameters        parameter to pass at the Krylov
                                       solver

            - auxiliary_sp             auxiliary parameters for setting
                                       solvers of inner blocks

            - v_d                      when solving non-linear problems,
                                       v_d is the non-linear residual
                                       (adjoint equation)

            - f                        when solving non-linear problems, f
                                       is the non-linear residual (state
                                       equation)

            - print_error              if True, the L^2 discrepancy between
                                       the desired state and the numerical
                                       solution is printed

            - create_output            if True, output is generated

            - plots                    if True, plots of the solutions are
                                       generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

        if any((not isinstance(bc.function_arg, ufl.classes.Zero)) for bc in self._bcs_v):
            v_inhom = Function(self.space_v)
            apply_bcs(self._bcs_v, v_inhom)
            bcs_v = homogenize(self._bcs_v)
        else:
            v_inhom = None
            bcs_v = self._bcs_v
        bcs_zeta = bcs_v

        nullspace_v = DirichletBCNullspace(bcs_v)
        nullspace_zeta = DirichletBCNullspace(bcs_zeta)

        v_old = Function(self.space_v, name="v_old").assign(self._v)
        D_v = self.construct_D_v(v_trial, v_test, v_old)
        D_zeta = adjoint(D_v)

        if f is None:
            f = self.construct_f(v_test, D_v, bcs_v, v_inhom=v_inhom)
        if v_d is None:
            v_d = self.construct_v_d(v_test, bcs_v, v_inhom=v_inhom)
        if solver_parameters is None:
            solver_parameters = {"linear_solver": "gmres",
                                 "gmres_restart": 10,
                                 "maximum_iterations": 50,
                                 "relative_tolerance": 1.0e-6,
                                 "absolute_tolerance": 0.0,
                                 "monitor_convergence": True}
        if P is None:
            pc_fn = self.construct_pc(auxiliary_sp, bcs_v, bcs_zeta, D_v, D_zeta)
        else:
            pc_fn = P(self, D_zeta, D_v, bcs_v, bcs_zeta)

        block_00 = {}
        block_00[(0, 0)] = self._M_v
        block_01 = {}
        block_01[(0, 0)] = D_zeta
        block_10 = {}
        block_10[(0, 0)] = D_v
        block_11 = {}
        block_11[(0, 0)] = -(1.0 / self.beta) * self._M_zeta
        system = MultiBlockSystem(
            self.space_v, self.space_v,
            block_00=block_00, block_01=block_01,
            block_10=block_10, block_11=block_11,
            nullspace_0=(nullspace_v,), nullspace_1=(nullspace_zeta,))

        v = Function(self.space_v, name="v")
        zeta = Function(self.space_v, name="zeta")
        system.solve(
            v, zeta, v_d, f,
            solver_parameters=solver_parameters,
            pc_fn=pc_fn)
        if v_inhom is not None:
            v += v_inhom
        self.set_v(v)
        self.set_zeta(zeta)

        del system, pc_fn

        if create_output:
            output({"v": v, "zeta": zeta})
        if plots:
            plot(v, zeta, self._true_v)
        if print_error:
            self.print_error()

    @garbage_cleanup_method()
    def non_linear_solve(self, *,
                         P=None, solver_parameters=None,
                         auxiliary_sp={},
                         max_non_linear_iter=10,
                         relative_non_linear_tol=1.0e-5,
                         absolute_non_linear_tol=1.0e-8,
                         print_error_non_linear=True,
                         create_output=True,
                         plots=False):
        """Module for the solution of non-linear control problems.

        Input:
            - P                           preconditioner to apply within
                                          the Krylov method (if None,
                                          default option is employed)

            - solver_parameters           parameter to pass at the Krylov
                                          solver

            - auxiliary_sp                auxiliary parameters for setting
                                          solvers of inner blocks

            - max_non_linear_iter         maximum number of non-linear
                                          iteration

            - relative_non_linear_tol     relative non-linear tolerance

            - absolute_non_linear_tol     absolute non-linear tolerance

            - print_error_non_linear      if True, the L^2 discrepancy
                                          between the desired state and the
                                          numerical solution is printed

            - create_output               if True, output is generated

            - plots                       if True, plots of the solutions
                                          are generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

        if any(not isinstance(bc.function_arg, ufl.classes.Zero) for bc in self._bcs_v):
            bcs_v = homogenize(self._bcs_v)
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v
            bcs_v_help = None
        bcs_zeta = bcs_v

        v_old = Function(self.space_v, name="v_old")
        zeta_old = Function(self.space_v, name="zeta_old")
        delta_v = Function(self.space_v, name="delta_v")
        delta_zeta = Function(self.space_v, name="delta_zeta")

        v_old.assign(self._v)
        zeta_old.assign(self._zeta)

        # construction of the discretized forward and adjoint forms
        D_v = self.construct_D_v(
            v_trial, v_test, v_old, non_linear_res=True)
        D_zeta = adjoint(D_v)
        M_zeta = -(1.0 / self.beta) * self._M_zeta

        # construction of the force function and the
        # desired state
        f = assemble(self._force_function(v_test))

        v_d, true_v = self._desired_state(v_test)
        v_d = assemble(v_d)
        self._true_v = true_v

        # construction of the non-linear residual
        rhs_0, rhs_1 = self.non_linear_res_eval(
            self.space_v, v_d, f, v_old, zeta_old,
            D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

        rhs = Cofunction((self.space_v * self.space_v).dual(), name="rhs")
        rhs.sub(0).assign(rhs_0)
        rhs.sub(1).assign(rhs_1)

        # initial norm of non-linear residual
        norm_0 = vnorm(rhs)
        norm_k = norm_0

        k = 0

        print(f'Initial non-linear residual: {norm_0:.16e}')

        while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):
            # solving the linearization
            self.linear_solve(P=P, solver_parameters=solver_parameters,
                              auxiliary_sp=auxiliary_sp,
                              v_d=rhs_0, f=rhs_1,
                              print_error=False,
                              create_output=False,
                              plots=False)

            delta_v.assign(self._v)
            delta_zeta.assign(self._zeta)

            # updating the state solution
            v_old += delta_v
            if bcs_v_help is not None:
                apply_bcs(bcs_v_help, v_old)
            self.set_v(v_old)

            # updating the adjoint solution
            zeta_old += delta_zeta
            apply_bcs(bcs_zeta, zeta_old)
            self.set_zeta(zeta_old)

            # construction of the discretized forward and adjoint forms
            D_v = self.construct_D_v(
                v_trial, v_test, v_old, non_linear_res=True)
            D_zeta = adjoint(D_v)

            # construction of the non-linear residual
            rhs_0, rhs_1 = self.non_linear_res_eval(
                self.space_v, v_d, f, v_old, zeta_old,
                D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

            rhs.sub(0).assign(rhs_0)
            rhs.sub(1).assign(rhs_1)

            # norm of non-linear residual
            norm_k = vnorm(rhs)

            k += 1

            print(f'Non-linear solver: '
                  f'iteration {k:d}, '
                  f'non-linear residual norm {norm_k:.16e}')

            if k + 1 > max_non_linear_iter:
                break

        if print_error_non_linear:
            if norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol:
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
                print(f'Number of non-linear iterations: {k:d}')
            else:
                print('The non-linear iteration did not converge')
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
            self.print_error()

        if create_output:
            output({"v": self._v, "zeta": self._zeta})

        if plots:
            plot(self._v, self._zeta, self._true_v)

    @garbage_cleanup_method()
    def incompressible_linear_solve(self, nullspace_p, *, space_p=None,
                                    P=None, solver_parameters=None,
                                    auxiliary_sp={},
                                    v_d=None, f=None,
                                    div_v=None, div_zeta=None,
                                    print_error=True, create_output=True,
                                    plots=False):
        """Module for the solution of linear incompressible control problems.

        Input:
            - nullspace_p              nullspace of the corresponding
                                       forward stationary incompressible
                                       problem

            - space_p                  pressure space, if not passed to the
                                       constructor

            - P                        preconditioner to apply within
                                       the Krylov method (if None, default
                                       option is employed)

            - solver_parameters        parameter to pass at the Krylov
                                       solver

            - auxiliary_sp             auxiliary parameters for setting
                                       solvers of inner blocks

            - v_d                      when solving non-linear problems,
                                       v_d is the non-linear residual
                                       (adjoint equation)

            - f                        when solving non-linear problems, f
                                       is the non-linear residual (state
                                       equation)

            - div_v                    when solving non-linear problems,
                                       div_v is the non-linear residual
                                       (incompressibility constraint on
                                       state variable)

            - div_zeta                 when solving non-linear problems,
                                       div_zeta is the non-linear residual
                                       (incompressibility constraint on
                                       adjoint variable)

            - print_error              if True, the L^2 discrepancy between
                                       the desired state and the numerical
                                       solution is printed

            - create_output            if True, output is generated

            - plots                    if True, plots of the solutions are
                                       generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)
        if space_p is None:
            if self._space_p is not None:
                space_p = self._space_p
            else:
                raise ValueError("Undefined space_p")
        else:
            self.set_space_p(space_p)
        p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

        if any(not isinstance(bc.function_arg, ufl.classes.Zero) for bc in self._bcs_v):
            bcs_v = homogenize(self._bcs_v)
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v
            bcs_v_help = None
        bcs_zeta = bcs_v

        # construction of nullspaces
        nullspace_v = DirichletBCNullspace(bcs_v)
        nullspace_zeta = DirichletBCNullspace(bcs_zeta)

        nullspace_0 = (nullspace_v, nullspace_zeta)
        nullspace_1 = (nullspace_p, nullspace_p)

        # construction of auxiliary spaces
        space_0 = FunctionSpace(
            self.space_v.mesh(), self.space_v.ufl_element() * self.space_v.ufl_element())
        space_1 = FunctionSpace(
            space_p.mesh(), space_p.ufl_element() * space_p.ufl_element())

        v_old = Function(self.space_v, name="v_old")
        v_old.assign(self._v)

        # construction of discretized forward and adjoint operators
        M_zeta = -(1.0 / self.beta) * self._M_zeta
        D_v = self.construct_D_v(v_trial, v_test, v_old)
        D_zeta = adjoint(D_v)

        B = - inner(div(v_trial), p_test) * dx
        B_T = - inner(p_trial, div(v_test)) * dx

        if bcs_v_help is not None:
            v_inhom = Function(self.space_v)
            apply_bcs(bcs_v_help, v_inhom)
        else:
            v_inhom = None

        # construction of force function
        if f is None:
            f = self.construct_f(v_test, D_v, bcs_v, v_inhom=v_inhom)

        # construction of desired state
        if v_d is None:
            v_d = self.construct_v_d(v_test, bcs_v, v_inhom=v_inhom)

        # construction of right-hand side
        if div_v is None:
            div_v = Function(space_p)
            if v_inhom is not None:
                div_v = assemble(- action(B, v_inhom))

        if div_zeta is None:
            div_zeta = Cofunction(space_p.dual())

        b_0 = Cofunction(space_0.dual(), name="b_0")
        b_1 = Cofunction(space_1.dual(), name="b_1")

        b_0.sub(0).assign(v_d)
        b_0.sub(1).assign(f)
        b_1.sub(0).assign(div_v)
        b_1.sub(1).assign(div_zeta)

        # construction of the system to be solved
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

        # construction of the linear system
        system = MultiBlockSystem(
            self.space_v, space_p,
            block_00=block_00, block_01=block_01,
            block_10=block_10, block_11=block_11,
            n_blocks_00=2, n_blocks_11=2,
            nullspace_0=nullspace_0, nullspace_1=nullspace_1)

        # construction of the preconditioner
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

            # solver parameters for pressure stiffness matrix
            if "sp_K_p" in auxiliary_sp:
                sp_K_p = auxiliary_sp["sp_K_p"]
            else:
                sp_K_p = {"ksp_type": "preonly",
                          "pc_type": "hypre",
                          "pc_hypre_type": "boomeramg",
                          "ksp_max_it": 1,
                          "pc_hypre_boomeramg_max_iter": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

            # solver parameters for pressure mass matrix
            if "sp_M_p" in auxiliary_sp:
                sp_M_p = auxiliary_sp["sp_M_p"]
            else:
                sp_M_p = {"ksp_type": "preonly",
                          "pc_type": "jacobi",
                          "ksp_max_it": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

            # building solvers for pressure stiffness and mass matrices
            solver_K_p = LinearSolver(
                assemble(K_p),
                solver_parameters=sp_K_p)

            solver_K_p.ksp.addConvergenceTest(converged, prepend=True)

            solver_M_p = LinearSolver(
                assemble(M_p), solver_parameters=sp_M_p)

            solver_M_p.ksp.addConvergenceTest(converged, prepend=True)

            if self._M_p is not None:
                block_00_p = self._M_p
            else:
                block_00_p = inner(p_trial, p_test) * dx
            block_10_p = self.construct_D_v(
                p_trial, p_test, v_old, non_linear_res=True)
            block_01_p = adjoint(block_10_p)
            if self._M_mu is not None:
                block_11_p = - (1.0 / self.beta) * self._M_mu
            else:
                block_11_p = - (1.0 / self.beta) * inner(p_trial, p_test) * dx

            # construction of inner system (coupled velocities)
            self._inner_system = MultiBlockSystem(
                self.space_v, self.space_v,
                block_00=block_00_int, block_01=block_01_int,
                block_10=block_10_int, block_11=block_11_int,
                nullspace_0=(nullspace_v,),
                nullspace_1=(nullspace_zeta,))

            # construction of inner preconditioner (coupled velocities)
            self._inner_pc_fn = self.construct_pc(
                auxiliary_sp, bcs_v, bcs_zeta, D_v, D_zeta)

            # construction of preconditioner for the whole system
            @garbage_cleanup(self.comm)
            def pc_fn(u_0, u_1, b_0, b_1):
                b_0_help = Cofunction(self.space_v.dual())
                b_1_help = Cofunction(self.space_v.dual())

                b_0_help.assign(b_0.sub(0))
                b_1_help.assign(b_0.sub(1))

                # solver parameters for inner solver
                if "sp_inner" in auxiliary_sp:
                    inner_solver_parameters = auxiliary_sp["sp_inner"]
                else:
                    inner_solver_parameters = {
                        "preconditioner": True,
                        "linear_solver": "gmres",
                        "maximum_iterations": 5,
                        "relative_tolerance": 0.0,
                        "absolute_tolerance": 0.0,
                        "monitor_convergence": False}

                v_help = Function(self.space_v)
                zeta_help = Function(self.space_v)

                # solver for the (1,1)-block
                _ = self._inner_system.solve(
                    v_help, zeta_help, b_0_help, b_1_help,
                    solver_parameters=inner_solver_parameters,
                    pc_fn=self._inner_pc_fn)

                u_0.sub(0).assign(v_help)
                u_0.sub(1).assign(zeta_help)

                # u_1 = - b_1 + block_10 * u_0
                b_0_help = Cofunction(space_p.dual())
                b_1_help = Cofunction(space_p.dual())
                b_0_help.assign(assemble(action(B, v_help) - b_1.sub(0)))
                b_1_help.assign(assemble(action(B, zeta_help) - b_1.sub(1)))

                # solving for the Schur complement approximation (apply
                # block-pressure convection--diffusion preconditioner)
                u_1.sub(0).zero()
                solver_K_p.solve(u_1.sub(0),
                                 b_0_help.copy(deepcopy=True))

                u_1.sub(1).zero()
                solver_K_p.solve(u_1.sub(1),
                                 b_1_help.copy(deepcopy=True))

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

                u_1.sub(0).zero()
                solver_M_p.solve(u_1.sub(0),
                                 b_0_help.copy(deepcopy=True))

                u_1.sub(1).zero()
                solver_M_p.solve(u_1.sub(1),
                                 b_1_help.copy(deepcopy=True))
        else:
            pc_fn = P(self, D_zeta, D_v, B,
                      nullspace_v, nullspace_zeta,
                      bcs_v, bcs_zeta)

        # solver parameters for the whole system
        if solver_parameters is None:
            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 50,
                                 "relative_tolerance": 1.0e-6,
                                 "absolute_tolerance": 0.0,
                                 "monitor_convergence": True}

        u_0_sol = Function(space_0)
        u_1_sol = Function(space_1)

        # solving linear system
        system.solve(
            u_0_sol, u_1_sol, b_0, b_1,
            solver_parameters=solver_parameters,
            pc_fn=pc_fn)

        v = Function(self.space_v, name="v")
        zeta = Function(self.space_v, name="zeta")
        p = Function(space_p, name="p")
        mu = Function(space_p, name="mu")

        v.assign(u_0_sol.sub(0))
        zeta.assign(u_0_sol.sub(1))

        # applying boundary conditions on state variable
        if v_inhom is not None:
            v += v_inhom

        p.assign(u_1_sol.sub(1))
        mu.assign(u_1_sol.sub(0))

        # updating solutions
        self.set_v(v)
        self.set_zeta(zeta)

        self.set_p(p)
        self.set_mu(mu)

        del system, pc_fn
        if P is None:
            del self._inner_system, self._inner_pc_fn

        if create_output:
            output({"v": v, "zeta": zeta, "p": p, "mu": mu})

        if plots:
            plot(v, p, zeta, mu, self._true_v)

        if print_error:
            self.print_error()

    @garbage_cleanup_method()
    def incompressible_non_linear_solve(self, nullspace_p, *, space_p=None,
                                        P=None, solver_parameters=None,
                                        auxiliary_sp={},
                                        max_non_linear_iter=10,
                                        relative_non_linear_tol=1.0e-5,
                                        absolute_non_linear_tol=1.0e-8,
                                        print_error_non_linear=True,
                                        create_output=True,
                                        plots=False):
        """Module for the solution of non-linear incompressible control problems.

        Input:
            - nullspace_p                nullspace of the corresponding
                                         forward stationary incompressible
                                         problem

            - space_p                    pressure space, if not passed to
                                         the constructor

            - P                          preconditioner to apply within
                                         the Krylov method (if None,
                                         default option is employed)

            - solver_parameters          parameter to pass at the Krylov
                                         solver

            - auxiliary_sp               auxiliary parameters for setting
                                         solvers of inner blocks

            - max_non_linear_iter        maximum number of non-linear
                                         iteration

            - relative_non_linear_tol    relative non-linear tolerance

            - absolute_non_linear_tol    absolute non-linear tolerance

            - print_error_non_linear     if True, the L^2 discrepancy
                                         between the desired state and the
                                         numerical solution is printed

            - create_output              if True, output is generated

            - plots                      if True, plots of the solutions
                                         are generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)
        if space_p is None:
            if self._space_p is not None:
                space_p = self._space_p
            else:
                raise ValueError("Undefined space_p")
        else:
            self.set_space_p(space_p)
        p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)

        # construction of auxiliary spaces
        space_0 = FunctionSpace(
            self.space_v.mesh(), self.space_v.ufl_element() * self.space_v.ufl_element())
        space_1 = FunctionSpace(
            space_p.mesh(), space_p.ufl_element() * space_p.ufl_element())

        if any(not isinstance(bc.function_arg, ufl.classes.Zero) for bc in self._bcs_v):
            bcs_v = homogenize(self._bcs_v)
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v
            bcs_v_help = None
        bcs_zeta = bcs_v

        v_old = Function(self.space_v, name="v_old")
        zeta_old = Function(self.space_v, name="zeta_old")
        delta_v = Function(self.space_v, name="delta_v")
        delta_zeta = Function(self.space_v, name="delta_zeta")

        p_old = Function(space_p, name="p_old")
        mu_old = Function(space_p, name="mu_old")
        delta_p = Function(space_p, name="delta_p")
        delta_mu = Function(space_p, name="delta_mu")

        v_old.assign(self._v)
        zeta_old.assign(self._zeta)
        p_old.assign(self._p)
        mu_old.assign(self._mu)

        # construction of discretized forward and adjoint forms
        D_v = self.construct_D_v(
            v_trial, v_test, v_old, non_linear_res=True)
        D_zeta = adjoint(D_v)
        M_zeta = -(1.0 / self.beta) * self._M_zeta

        B = - inner(div(v_trial), p_test) * dx
        B_T = - inner(p_trial, div(v_test)) * dx

        # construction of force function and desired state
        f = assemble(self._force_function(v_test))

        v_d, true_v = self._desired_state(v_test)
        v_d = assemble(v_d)
        self._true_v = true_v

        # function for the evaluation of the non-linear residual,
        # in case of incompressible control problems
        @garbage_cleanup(self.comm)
        def non_linear_res_eval():
            rhs_0, rhs_1 = self.non_linear_res_eval(
                self.space_v, v_d, f, v_old, zeta_old,
                D_v, D_zeta, M_zeta, bcs_v, bcs_zeta)

            rhs_00 = assemble(rhs_0 - action(B_T, mu_old))
            rhs_01 = assemble(rhs_1 - action(B_T, p_old))
            rhs_10 = assemble(-action(B, v_old))
            rhs_11 = assemble(-action(B, zeta_old))
            apply_bcs(bcs_v, rhs_00)
            apply_bcs(bcs_zeta, rhs_01)

            return rhs_00, rhs_01, rhs_10, rhs_11

        # construction of the non-linear residual
        rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

        rhs = Cofunction((space_0 * space_1).dual(), name="rhs")
        rhs.sub(0).assign(rhs_00)
        rhs.sub(1).assign(rhs_01)
        rhs.sub(2).assign(rhs_10)
        rhs.sub(3).assign(rhs_11)

        # initial norm of the non-linear residual
        norm_0 = vnorm(rhs)
        norm_k = norm_0

        k = 0

        print(f'Initial non-linear residual: {norm_0:.16e}')

        while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):
            # solving for the linearization
            self.incompressible_linear_solve(
                nullspace_p, space_p=space_p, P=P,
                solver_parameters=solver_parameters,
                auxiliary_sp=auxiliary_sp,
                v_d=rhs_00, f=rhs_01,
                div_v=rhs_10, div_zeta=rhs_11,
                print_error=False, create_output=False, plots=False)

            delta_v.assign(self._v)
            delta_zeta.assign(self._zeta)
            delta_p.assign(self._p)
            delta_mu.assign(self._mu)

            # updating the solutions
            v_old += delta_v
            if bcs_v_help is not None:
                apply_bcs(bcs_v_help, v_old)
            self.set_v(v_old)

            zeta_old += delta_zeta
            apply_bcs(bcs_zeta, zeta_old)
            self.set_zeta(zeta_old)

            p_old += delta_p
            self.set_p(p_old)

            mu_old += delta_mu
            self.set_mu(mu_old)

            # construction of the discretized forward and adjoint forms
            D_v = self.construct_D_v(
                v_trial, v_test, v_old, non_linear_res=True)
            D_zeta = adjoint(D_v)

            # construction of the non-linear residual
            rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

            rhs.sub(0).assign(rhs_00)
            rhs.sub(1).assign(rhs_01)
            rhs.sub(2).assign(rhs_10)
            rhs.sub(3).assign(rhs_11)

            # norm of the non-linear residual
            norm_k = vnorm(rhs)

            k += 1

            print(f'Non-linear solver: '
                  f'iteration {k:d}, '
                  f'non-linear residual norm {norm_k:.16e}')

            if k + 1 > max_non_linear_iter:
                break

        if print_error_non_linear:
            if norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol:
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
                print(f'Number of non-linear iterations: {k:d}')
            else:
                print('The non-linear iteration did not converge')
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
            self.print_error()

        if create_output:
            output({"v": self._v, "zeta": self._zeta, "p": self._p, "mu": self._mu})

        if plots:
            plot(self._v, self._p, self._zeta, self._mu, self._true_v)


class Instationary:
    def __init__(self, space_v, forward_form, desired_state=None,
                 force_function=None, *, beta=1.0e-3, space_p=None,
                 Gauss_Newton=False, CN=True, n_t=20,
                 initial_condition=None, time_interval=(0.0, 1.0), bcs_v=None):
        """Instationary control problem.

        Input:
            - space_v             space to which the solution belongs

            - forward_form        form that represents the differential
                                  operator in space

            - desired_state       desired state, defaults to zero

            - force_function      force function acting on the system,
                                  defaults to zero

            - beta                regularization parameter

            - space_p             pressure space (only for incompressible
                                  problems)

            - Gauss_Newton        if True, a Gauss--Newton linearization is
                                  employed, otherwise a Picard
                                  linearization is applied

            - CN                  if True, the trapezoidal rule is employed
                                  as discretization in time

            - n_t                 number of points in time

            - initial_condition   initial condition

            - time_interval       interval of time itegration

            - bcs_v               boundary conditions on the state
        """

        bcs_v_arg = bcs_v

        def bcs_v(space_v, t):
            if bcs_v_arg is None:
                bc_v = ()
            else:
                bc_v = bcs_v_arg(space_v, t)
            if not isinstance(bc_v, Sequence):
                bc_v = (bc_v,)
            return tuple(bc_v)

        if not isinstance(space_v, FunctionSpaceBase):
            raise TypeError("Space must be a primal space")
        if space_p is not None \
                and not isinstance(space_p, FunctionSpaceBase):
            raise TypeError("Space must be a primal space")

        if desired_state is None:
            def desired_state(test_v, t):
                return ZeroBaseForm((test_v,))
        if force_function is None:
            def force_function(test_v, t):
                return ZeroBaseForm((test_v,))

        self._space_v = space_v
        self._forward_form = forward_form
        self._desired_state = desired_state
        self._force_function = force_function
        self._beta = beta
        self._Gauss_Newton = Gauss_Newton
        self._CN = CN
        self._n_t = n_t
        self._initial_condition = initial_condition
        self._time_interval = time_interval

        self._f_bcs_v = bcs_v
        self._bcs_v = {i: bcs_v(space_v, Constant(time(time_interval, i, n_t)))
                       for i in range(n_t)}

        flattened_space_v = tuple(space_v for _ in range(n_t))
        full_space_v = MixedFunctionSpace(flattened_space_v)
        self._v = Function(full_space_v, name="v")
        self._zeta = Function(full_space_v, name="zeta")
        for i in range(n_t):
            apply_bcs(self._bcs_v[i], self._v.sub(i))

        v_test, v_trial = TestFunction(space_v), TrialFunction(space_v)
        self._M_v = self._M_zeta = inner(v_trial, v_test) * dx

        if space_p is not None:
            self.set_space_p(space_p)
        else:
            self._space_p = None
            self._M_p = self._M_mu = None

    @property
    def space_v(self):
        return self._space_v

    @cached_property
    def comm(self):
        return self.space_v.mesh().comm

    @property
    def beta(self):
        return self._beta

    def time(self, i):
        return time(self._time_interval, i, self._n_t)

    def set_space_p(self, space_p):
        """
        Input:
            - space_p        new pressure space
        """

        if not self._CN:
            flattened_space_p = tuple(space_p for _ in range(self._n_t))
        else:
            flattened_space_p = tuple(space_p for _ in range(self._n_t - 1))
        full_space_p = MixedFunctionSpace(flattened_space_p)

        self._space_p = space_p
        self._p = Function(full_space_p, name="p")
        self._mu = Function(full_space_p, name="mu")
        p_test, p_trial = TestFunction(space_p), TrialFunction(space_p)
        self._M_p = self._M_mu = inner(p_trial, p_test) * dx

    def set_v(self, v_new):
        """
        Input:
            - v_new        new approximation of the state solution
        """

        self._v.assign(v_new)
        for i in range(self._n_t):
            apply_bcs(self._bcs_v[i], self._v.sub(i))

    def set_zeta(self, zeta_new):
        """
        Input:
            - zeta_new        new approximation of the adjoint solution
        """

        self._zeta.assign(zeta_new)
        for i in range(self._n_t):
            apply_bcs(homogenize(self._bcs_v[1]), self._zeta.sub(i))

    def set_p(self, p_new):
        """
        Input:
            - p_new        new approximation of the pressure state solution
        """

        self._p.assign(p_new)

    def set_mu(self, mu_new):
        """
        Input:
            - mu_new        new approximation of the pressure adjoint
                            solution
        """

        self._mu.assign(mu_new)

    def print_error(self, tau):
        """Print L^2 norm of the difference between the numerical solution
        and the desired state.
        """

        error_norm = sqrt(tau) * norm(self._true_v - self._v, norm_type="L2")
        print(f'Estimated error in the L2-norm: {error_norm:.16e}')

    def construct_D_v(self, v_trial, v_test, v_n_help, t, *,
                      non_linear_res=False):
        """Construction of the discretized forward form.

        Input:
            - v_trial               trial function

            - v_test                test function

            - v_n_help              approximation of the state solution at
                                    time t

            - t                     time point in which evaluating the form

            - non_linear_res        if True, the form is employed in the
                                    evaluation of the non-linear residual

        Output:
            - D_v                   discretized forward form
        """

        if (not self._Gauss_Newton) or non_linear_res:
            # if Gauss--Newton is not applied or we want to
            # evaluate the residual, we take the Picard linearization
            # of the forward form
            D_v_i = self._forward_form(v_trial, v_test, v_n_help, t)
        else:
            # if we want to apply Gauss--Newton, we take the
            # derivative of the form in the direction of v_n_help
            D_v_i = ufl.derivative(
                self._forward_form(v_n_help, v_test, v_n_help, t),
                v_n_help, v_trial)

        return D_v_i

    def construct_f(self, full_space_v, v_test):
        """Construction of the vector containing the force function.

        Input:
            - full_space_v        full space for time integration

            - v_test              test function

        Output:
            - f                   discretized force function
        """

        f = Cofunction(full_space_v.dual(), name="f")
        for i in range(self._n_t):
            f.sub(i).assign(
                assemble(self._force_function(v_test, Constant(self.time(i)))))
        return f

    def construct_v_d(self, full_space_v, v_test):
        """Construction of the vector containing the desired state.

        Input:
            - full_space_v        full space for time integration

            - v_test              test function

        Output:
            - v_d                 discretized desired state
        """

        v_d = Cofunction(full_space_v.dual(), name="v_d")
        true_v = Function(full_space_v, name="true_v")
        for i in range(self._n_t):
            v_d_i, true_v_i = self._desired_state(v_test, Constant(self.time(i)))
            v_d.sub(i).assign(assemble(v_d_i))
            true_v.sub(i).assign(true_v_i)

        self._true_v = true_v
        return v_d

    def construct_pc(self, auxiliary_sp, full_space_v,
                     bcs_v, bcs_zeta, block_01, block_10, epsilon=None):
        """Construction of the preconditioner, based on the matching strategy.

        Input:
            - auxiliary_sp        auxiliary solver parameters for inner
                                  blocks

            - full_space_v        full space for time integration

            - bcs_v               homogenized boundary conditions for the
                                  state variable

            - bcs_zeta            homogenized boundary conditions for the
                                  adjoint variable

            - block_01            (1,2)-block of the linear system,
                                  containing discretized adjoint forms

            - block_10            (2,1)-block of the linear system,
                                  containing discretized state forms

            - epsilon             parameters employed for the construction
                                  of the preconditioner for the BE
                                  discretization

        Output:
            - pc_linear           preconditioner to employ within Krylov
                                  method
        """

        n_t = self._n_t
        t_0 = self._time_interval[0]
        T_f = self._time_interval[1]

        tau = (T_f - t_0) / (n_t - 1.0)

        # solver parameters for the (1,1)-block
        if "sp_11block" in auxiliary_sp:
            sp_11block = auxiliary_sp["sp_11block"]
        else:
            sp_11block = {"ksp_type": "preonly",
                          "pc_type": "jacobi",
                          "ksp_max_it": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

        # solver parameters for the factorization of the Schur complement
        # approximation
        if "sp_Schur" in auxiliary_sp:
            sp_Schur = auxiliary_sp["sp_Schur"]
        else:
            sp_Schur = {"ksp_type": "preonly",
                        "pc_type": "hypre",
                        "pc_hypre_type": "boomeramg",
                        "ksp_max_it": 1,
                        "pc_hypre_boomeramg_max_iter": 2,
                        "ksp_atol": 0.0,
                        "ksp_rtol": 0.0}

        # constructing the solver for the (1,1)-block
        solver_0 = LinearSolver(
            assemble(self._M_v, bcs=bcs_v),
            solver_parameters=sp_11block)

        solver_0.ksp.addConvergenceTest(converged, prepend=True)

        solver_state = {}
        solver_adj = {}

        if self._CN:
            # building the solvers for the preconditioner for the
            # trapezoidal rule
            my_const = Constant(0.5 * tau / (self.beta**0.5))

            for i in range(n_t - 1):
                block_ii = block_10[(i, i)]
                solver_i_state = LinearSolver(
                    assemble(block_ii + my_const * self._M_v,
                             bcs=bcs_zeta),
                    solver_parameters=sp_Schur)
                solver_i_state.ksp.addConvergenceTest(
                    converged, prepend=True)
                solver_state[(i)] = solver_i_state

                block_ii = block_01[(i, i)]
                solver_i_adj = LinearSolver(
                    assemble(block_ii
                             + my_const * self._M_zeta,
                             bcs=bcs_zeta),
                    solver_parameters=sp_Schur)
                solver_i_adj.ksp.addConvergenceTest(
                    converged, prepend=True)
                solver_adj[(i)] = solver_i_adj
        else:
            # building the solvers for the preconditioner for
            # backward Euler
            my_const = Constant(tau / (self.beta**0.5))

            block_ii = block_10[(0, 0)]
            solver_i_state = LinearSolver(
                assemble(block_ii, bcs=bcs_zeta),
                solver_parameters=sp_Schur)
            solver_i_state.ksp.addConvergenceTest(converged, prepend=True)
            solver_state[(0)] = solver_i_state

            block_ii = block_01[(0, 0)]
            solver_i_adj = LinearSolver(
                assemble(block_ii, bcs=bcs_zeta),
                solver_parameters=sp_Schur)
            solver_i_adj.ksp.addConvergenceTest(converged, prepend=True)
            solver_adj[(0)] = solver_i_adj

            for i in range(1, n_t - 1):
                block_ii = block_10[(i, i)]
                solver_i_state = LinearSolver(
                    assemble(block_ii + my_const * self._M_v,
                             bcs=bcs_zeta),
                    solver_parameters=sp_Schur)
                solver_i_state.ksp.addConvergenceTest(
                    converged, prepend=True)
                solver_state[(i)] = solver_i_state

                block_ii = block_01[(i, i)]
                solver_i_adj = LinearSolver(
                    assemble(block_ii + my_const * self._M_v,
                             bcs=bcs_zeta),
                    solver_parameters=sp_Schur)
                solver_i_adj.ksp.addConvergenceTest(
                    converged, prepend=True)
                solver_adj[(i)] = solver_i_adj

            block_ii = block_10[(n_t - 1, n_t - 1)]
            solver_i_state = LinearSolver(
                assemble(block_ii + (epsilon**0.5) * my_const * self._M_v,
                         bcs=bcs_zeta),
                solver_parameters=sp_Schur)
            solver_i_state.ksp.addConvergenceTest(converged, prepend=True)
            solver_state[(n_t - 1)] = solver_i_state

            block_ii = block_01[(n_t - 1, n_t - 1)]
            solver_i_adj = LinearSolver(
                assemble(block_ii + (epsilon**0.5) * my_const * self._M_v,
                         bcs=bcs_zeta),
                solver_parameters=sp_Schur)
            solver_i_adj.ksp.addConvergenceTest(converged, prepend=True)
            solver_adj[(n_t - 1)] = solver_i_adj

        del solver_i_state, solver_i_adj

        # definition of preconditioner
        if self._CN:
            # preconditioner for the trapezoidal rule
            @garbage_cleanup(self.comm)
            def pc_linear(u_0, u_1, b_0, b_1):
                # solving for the (1,1)-block
                b_0_help = apply_T_1_inv(b_0, self.space_v, n_t - 1)

                for i in range(n_t - 1):
                    b = Cofunction(self.space_v.dual())
                    b.assign(b_0_help.sub(i))
                    u_0.sub(i).zero()
                    solver_0.solve(u_0.sub(i),
                                   b.copy(deepcopy=True))
                    with u_0.sub(i).dat.vec as x_v:
                        x_v.scale(2.0 / Constant(tau))

                b_0_help = apply_T_2_inv(u_0, self.space_v, n_t - 1)
                for i in range(n_t - 1):
                    u_0.sub(i).assign(b_0_help.sub(i))

                # u_1 = - b_1 + D_v * u_0
                b = Cofunction(full_space_v.dual())
                block_ii = block_10[(0, 0)]
                b_help = Function(self.space_v)
                b_help.assign(u_0.sub(0))
                b.sub(0).assign(assemble(action(block_ii, b_help)))
                apply_bcs(bcs_zeta, b.sub(0))

                for i in range(1, n_t - 1):
                    block_ii = block_10[(i, i)]
                    block_ij = block_10[(i, i - 1)]
                    b_help_i = Function(self.space_v)
                    b_help_i.assign(u_0.sub(i))
                    b_help_j = Function(self.space_v)
                    b_help_j.assign(u_0.sub(i - 1))
                    b.sub(i).assign(assemble(
                        action(block_ii, b_help_i)
                        + action(block_ij, b_help_j)))
                    apply_bcs(bcs_zeta, b.sub(i))

                b = apply_T_2(b, self.space_v, n_t - 1)

                b -= b_1
                for i in range(n_t - 1):
                    apply_bcs(bcs_zeta, b.sub(i))

                # solving for the Schur complement approximation
                b = apply_T_2_inv(b, self.space_v, n_t - 1)
                # first solve
                solver_1 = solver_state[(0)]
                b_help = Cofunction(self.space_v.dual())
                b_help.assign(b.sub(0))
                u_1.sub(0).zero()
                solver_1.solve(u_1.sub(0),
                               b_help.copy(deepcopy=True))

                for i in range(1, n_t - 1):
                    block_ij = block_10[(i, i - 1)] + my_const * self._M_v
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i - 1))
                    b_help_new = assemble(b.sub(i) - action(block_ij, b_help))
                    b.sub(i).assign(b_help_new)
                    apply_bcs(bcs_zeta, b.sub(i))
                    b_help = Cofunction(self.space_v.dual())
                    b_help.assign(b.sub(i))
                    solver_1 = solver_state[(i)]
                    u_1.sub(i).zero()
                    solver_1.solve(u_1.sub(i),
                                   b_help.copy(deepcopy=True))

                # apply T_2
                b = apply_T_2(u_1, self.space_v, n_t - 1)
                for i in range(n_t - 1):
                    u_1.sub(i).assign(b.sub(i))

                # mat-mult by blk(M_v,...,M_v)
                b = Cofunction(full_space_v.dual())
                for i in range(n_t - 1):
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i))
                    b.sub(i).assign(assemble(action(self._M_v,
                                                    b_help)))
                    with b.sub(i).dat.vec as b_v:
                        b_v.scale(0.5 * Constant(tau))
                    apply_bcs(bcs_zeta, b.sub(i))

                # second solve
                b_help = Cofunction(self.space_v.dual())
                b_help.assign(b.sub(n_t - 2))
                solver_2 = solver_adj[(n_t - 2)]
                u_1.sub(n_t - 2).zero()
                solver_2.solve(u_1.sub(n_t - 2),
                               b_help.copy(deepcopy=True))

                for i in range(n_t - 3, -1, -1):
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i + 1))
                    block_ij = block_01[(i, i + 1)] + my_const * self._M_zeta
                    b_help_new = assemble(b.sub(i) - action(block_ij, b_help))
                    b.sub(i).assign(b_help_new)
                    apply_bcs(bcs_zeta, b.sub(i))
                    b_help = Cofunction(self.space_v.dual())
                    b_help.assign(b.sub(i))
                    solver_2 = solver_adj[(i)]
                    u_1.sub(i).zero()
                    solver_2.solve(u_1.sub(i),
                                   b_help.copy(deepcopy=True))
        else:
            # preconditioner for backward Euler
            @garbage_cleanup(self.comm)
            def pc_linear(u_0, u_1, b_0, b_1):
                # solving for the (1,1)-block
                for i in range(n_t):
                    b = Cofunction(self.space_v.dual())
                    b.assign(b_0.sub(i))
                    u_0.sub(i).zero()
                    solver_0.solve(u_0.sub(i), b.copy(deepcopy=True))
                    with u_0.sub(i).dat.vec as x_v:
                        x_v.scale(1.0 / Constant(tau))

                with u_0.sub(n_t - 1).dat.vec as x_v:
                    x_v.scale(1.0 / epsilon)

                # u_1 = - b_1 + D_v * u_0
                b = Cofunction(full_space_v.dual())
                block_ii = block_10[(0, 0)]
                b_help = Function(self.space_v)
                b_help.assign(u_0.sub(0))
                b.sub(0).assign(assemble(action(block_ii, b_help)))

                for i in range(1, n_t):
                    block_ii = block_10[(i, i)]
                    block_ij = block_10[(i, i - 1)]
                    b_help_i = Function(self.space_v)
                    b_help_i.assign(u_0.sub(i))
                    b_help_j = Function(self.space_v)
                    b_help_j.assign(u_0.sub(i - 1))
                    b.sub(i).assign(assemble(
                        action(block_ii, b_help_i)
                        + action(block_ij, b_help_j)))

                b -= b_1
                for i in range(n_t):
                    apply_bcs(bcs_zeta, b.sub(i))

                # solving for the Schur complement approximation
                # first solve
                b_help = Cofunction(self.space_v.dual())
                b_help.assign(b.sub(0))
                solver_1 = solver_state[(0)]
                u_1.sub(0).zero()
                solver_1.solve(u_1.sub(0),
                               b_help.copy(deepcopy=True))

                for i in range(1, n_t):
                    block_ij = block_10[(i, i - 1)]
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i - 1))
                    b_help_new = assemble(b.sub(i) - action(block_ij, b_help))
                    b.sub(i).assign(b_help_new)
                    apply_bcs(bcs_zeta, b.sub(i))

                    b_help = Cofunction(self.space_v.dual())
                    b_help.assign(b.sub(i))
                    solver_1 = solver_state[(i)]
                    u_1.sub(i).zero()
                    solver_1.solve(u_1.sub(i),
                                   b_help.copy(deepcopy=True))

                # mat-mult by blk(M_v,...,M_v)
                b = Cofunction(full_space_v.dual())
                for i in range(n_t - 1):
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i))
                    b.sub(i).assign(assemble(action(self._M_v,
                                                    b_help)))
                    with b.sub(i).dat.vec as b_v:
                        b_v.scale(Constant(tau))
                    apply_bcs(bcs_zeta, b.sub(i))

                b_help = Function(self.space_v)
                b_help.assign(u_1.sub(n_t - 1))
                b.sub(n_t - 1).assign(assemble(action(self._M_v,
                                                      b_help)))
                with b.sub(n_t - 1).dat.vec as b_v:
                    b_v.scale(epsilon * Constant(tau))
                apply_bcs(bcs_zeta, b.sub(n_t - 1))

                # second solve
                b_help = Cofunction(self.space_v.dual())
                b_help.assign(b.sub(n_t - 1))
                solver_2 = solver_adj[(n_t - 1)]
                u_1.sub(n_t - 1).zero()
                solver_2.solve(u_1.sub(n_t - 1),
                               b_help.copy(deepcopy=True))

                for i in range(n_t - 2, -1, -1):
                    b_help = Function(self.space_v)
                    b_help.assign(u_1.sub(i + 1))
                    block_ij = block_01[(i, i + 1)]
                    b_help_new = assemble(b.sub(i) - action(block_ij, b_help))
                    b.sub(i).assign(b_help_new)
                    apply_bcs(bcs_zeta, b.sub(i))

                    b_help = Cofunction(self.space_v.dual())
                    b_help.assign(b.sub(i))
                    solver_2 = solver_adj[(i)]
                    u_1.sub(i).zero()
                    solver_2.solve(u_1.sub(i),
                                   b_help.copy(deepcopy=True))

        return pc_linear

    @garbage_cleanup_method()
    def non_linear_res_eval(self, full_space_v, v_old, zeta_old, v_0,
                            v_d, f, M_v, bcs_v, bcs_zeta):
        """Construction of the non-linear residual.

        Input:
            - full_space_v      full space for time integration

            - v_old             approximation of state variable

            - zeta_old          approximation of adjoint variable

            - v_0               initial condition on state variable

            - v_d               desired state

            - f                 force function

            - M_v               mass matrix on the state space

            - bcs_v             homogenized boundary conditions for the
                                state variable

            - bcs_zeta          homogenized boundary conditions for the
                                adjoint variable

        Output:
            - rhs_0             non-linear residual (adjoint equation)

            - rhs_1             non-linear residual (state equation)
        """

        n_t = self._n_t
        t_0 = self._time_interval[0]
        T_f = self._time_interval[1]

        tau = (T_f - t_0) / (n_t - 1.0)
        if not self._CN:
            const_tau = Constant(tau)
            const_tau_beta = Constant(tau / self.beta)
        else:
            const_tau = Constant(0.5 * tau)
            const_tau_beta = Constant(0.5 * tau / self.beta)

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

        rhs_0 = Cofunction(full_space_v.dual(), name="rhs_0")
        rhs_1 = Cofunction(full_space_v.dual(), name="rhs_1")

        if not self._CN:
            # evaluating the non-linear residual for backward Euler
            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_old.sub(0), Constant(t_0),
                                       non_linear_res=True)
            D_zeta_i = adjoint(D_v_i)

            D_v_0 = self.construct_D_v(v_trial, v_test,
                                       v_0, Constant(t_0),
                                       non_linear_res=True)

            b_help_v = Function(self.space_v)
            b_help_v.assign(v_old.sub(0))
            b_help_zeta0 = Function(self.space_v)
            b_help_zeta0.assign(zeta_old.sub(0))
            b_help_zeta1 = Function(self.space_v)
            b_help_zeta1.assign(zeta_old.sub(1))
            rhs_0.sub(0).assign(assemble(
                tau * v_d.sub(0) - action(const_tau * self._M_v, b_help_v)
                - action(const_tau * D_zeta_i + M_v, b_help_zeta0)
                + action(M_v, b_help_zeta1)))
            apply_bcs(bcs_zeta, rhs_0.sub(0))

            b_help = Function(self.space_v)
            b_help.assign(v_old.sub(0))
            rhs_1.sub(0).assign(assemble(
                action(const_tau * D_v_0 + M_v, v_0)
                - action(const_tau * D_v_i + M_v, b_help)))
            apply_bcs(bcs_v, rhs_1.sub(0))

            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_old.sub(n_t - 1), Constant(T_f),
                                       non_linear_res=True)
            D_zeta_i = adjoint(D_v_i)

            b_help_v0 = Function(self.space_v)
            b_help_v0.assign(v_old.sub(n_t - 2))
            b_help_v1 = Function(self.space_v)
            b_help_v1.assign(v_old.sub(n_t - 1))
            b_help_zeta = Function(self.space_v)
            b_help_zeta.assign(zeta_old.sub(n_t - 1))
            rhs_1.sub(n_t - 1).assign(assemble(
                tau * f.sub(n_t - 1) + action(M_v, b_help_v0)
                - action(const_tau * D_v_i + M_v, b_help_v1)
                + action(const_tau_beta * self._M_zeta, b_help_zeta)))
            apply_bcs(bcs_v, rhs_1.sub(n_t - 1))

            b_help = Function(self.space_v)
            b_help.assign(zeta_old.sub(n_t - 1))
            rhs_0.sub(n_t - 1).assign(assemble(
                - action(const_tau * D_zeta_i + M_v, b_help)))
            apply_bcs(bcs_zeta, rhs_0.sub(n_t - 1))

            t = t_0
            for i in range(1, n_t - 1):
                t += tau
                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_old.sub(i), Constant(t),
                                           non_linear_res=True)
                D_zeta_i = adjoint(D_v_i)

                b_help_v0 = Function(self.space_v)
                b_help_v0.assign(v_old.sub(i - 1))
                b_help_v1 = Function(self.space_v)
                b_help_v1.assign(v_old.sub(i))
                b_help_zeta0 = Function(self.space_v)
                b_help_zeta0.assign(zeta_old.sub(i))
                b_help_zeta1 = Function(self.space_v)
                b_help_zeta1.assign(zeta_old.sub(i + 1))

                rhs_0.sub(i).assign(assemble(
                    tau * v_d.sub(i) + action(M_v, b_help_zeta1)
                    - action(const_tau * self._M_v, b_help_v1)
                    - action(const_tau * D_zeta_i + M_v, b_help_zeta0)))
                apply_bcs(bcs_zeta, rhs_0.sub(i))

                rhs_1.sub(i).assign(assemble(
                    tau * f.sub(i) + action(M_v, b_help_v0)
                    - action(const_tau * D_v_i + M_v, b_help_v1)
                    + action(const_tau_beta * self._M_zeta, b_help_zeta0)))
                apply_bcs(bcs_v, rhs_1.sub(i))
        else:
            # evaluating non-linear residual for the trapezoidal rule
            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_old.sub(0), Constant(t_0),
                                       non_linear_res=True)
            D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                            v_old.sub(1),
                                            Constant(t_0 + tau),
                                            non_linear_res=True)
            D_zeta_i = adjoint(D_v_i)
            D_zeta_i_plus = adjoint(D_v_i_plus)

            b_help_v0 = Function(self.space_v)
            b_help_v0.assign(v_old.sub(0))
            b_help_v1 = Function(self.space_v)
            b_help_v1.assign(v_old.sub(1))
            b_help_zeta0 = Function(self.space_v)
            b_help_zeta0.assign(zeta_old.sub(0))
            b_help_zeta1 = Function(self.space_v)
            b_help_zeta1.assign(zeta_old.sub(1))

            rhs_0.sub(0).assign(assemble(
                0.5 * tau * (v_d.sub(0) + v_d.sub(1))
                - action(const_tau * self._M_v, b_help_v0)
                - action(const_tau * self._M_v, b_help_v1)
                - action(const_tau * D_zeta_i + M_v, b_help_zeta0)
                - action(const_tau * D_zeta_i_plus - M_v, b_help_zeta1)))
            apply_bcs(bcs_zeta, rhs_0.sub(0))

            rhs_1.sub(0).assign(assemble(
                0.5 * tau * (f.sub(0) + f.sub(1))
                - action(const_tau * D_v_i - M_v, b_help_v0)
                - action(const_tau * D_v_i_plus + M_v, b_help_v1)
                + action(const_tau_beta * self._M_zeta, b_help_zeta0)
                + action(const_tau_beta * self._M_zeta, b_help_zeta1)))
            apply_bcs(bcs_v, rhs_1.sub(0))

            t = t_0
            for i in range(1, n_t - 1):
                t += tau
                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_old.sub(i), Constant(t),
                                           non_linear_res=True)
                D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                v_old.sub(i + 1),
                                                Constant(t + tau),
                                                non_linear_res=True)
                D_zeta_i = adjoint(D_v_i)
                D_zeta_i_plus = adjoint(D_v_i_plus)

                b_help_v0 = Function(self.space_v)
                b_help_v0.assign(v_old.sub(i))
                b_help_v1 = Function(self.space_v)
                b_help_v1.assign(v_old.sub(i + 1))
                b_help_zeta0 = Function(self.space_v)
                b_help_zeta0.assign(zeta_old.sub(i))
                b_help_zeta1 = Function(self.space_v)
                b_help_zeta1.assign(zeta_old.sub(i + 1))

                rhs_0.sub(i).assign(assemble(
                    0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1))
                    - action(const_tau * self._M_v, b_help_v0)
                    - action(const_tau * self._M_v, b_help_v1)
                    - action(const_tau * D_zeta_i + M_v, b_help_zeta0)
                    - action(const_tau * D_zeta_i_plus - M_v, b_help_zeta1)))
                apply_bcs(bcs_zeta, rhs_0.sub(i))

                rhs_1.sub(i).assign(assemble(
                    0.5 * tau * (f.sub(i) + f.sub(i + 1))
                    - action(const_tau * D_v_i - M_v, b_help_v0)
                    - action(const_tau * D_v_i_plus + M_v, b_help_v1)
                    + action(const_tau_beta * self._M_zeta, b_help_zeta0)
                    + action(const_tau_beta * self._M_zeta, b_help_zeta1)))
                apply_bcs(bcs_v, rhs_1.sub(i))

        return rhs_0, rhs_1

    @garbage_cleanup_method()
    def linear_solve(self, *,
                     P=None, solver_parameters=None,
                     auxiliary_sp={}, v_d=None, f=None,
                     print_error=True, create_output=True,
                     plots=False):
        """Module for the solution of linear control problems.

        Input:
            - P                        preconditioner to apply within
                                       the Krylov method (if None, default
                                       option is employed)

            - solver_parameters        parameter to pass at the Krylov
                                       solver

            - auxiliary_sp             auxiliary parameters for setting
                                       solvers of inner blocks

            - v_d                      when solving non-linear problems,
                                       v_d is the non-linear residual
                                       (adjoint equation)

            - f                        when solving non-linear problems, f
                                       is the non-linear residual (state
                                       equation)

            - print_error              if True, the L^2 discrepancy between
                                       the desired state and the numerical
                                       solution is printed

            - create_output            if True, output is generated

            - plots                    if True, plots of the solutions are
                                       generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

        n_t = self._n_t
        t_0 = self._time_interval[0]
        T_f = self._time_interval[1]
        tau = (T_f - t_0) / (n_t - 1.0)
        if not self._CN:
            const_tau = Constant(tau)
            const_tau_beta = Constant(tau / self.beta)
        else:
            const_tau = Constant(0.5 * tau)
            const_tau_beta = Constant(0.5 * tau / self.beta)

        if not self._CN:
            epsilon = Constant(1.0e-3)

        inhomogeneous_bcs_v = False
        for i, bc_i in self._bcs_v.items():
            for bc in bc_i:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

        if inhomogeneous_bcs_v:
            bcs_v = homogenize(self._bcs_v[(1)])
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v[(1)]
        bcs_zeta = bcs_v

        # construction of nullspaces
        nullspace_v = DirichletBCNullspace(bcs_v)
        nullspace_zeta = DirichletBCNullspace(bcs_zeta)

        # construction of full space for time integration
        full_nullspace_v = ()
        full_nullspace_zeta = ()
        for i in range(n_t - 1):
            full_nullspace_v = full_nullspace_v + (nullspace_v, )
            full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )

        if not self._CN:
            full_nullspace_v = full_nullspace_v + (nullspace_v, )
            full_nullspace_zeta = full_nullspace_zeta + (nullspace_zeta, )

        flattened_space_v = tuple(self.space_v for i in range(n_t))
        full_space_v = MixedFunctionSpace(flattened_space_v)

        # construction of initial condition
        if self._initial_condition is not None:
            v_0 = self._initial_condition(v_test)
        else:
            v_0 = Function(self.space_v, name="v_0")

        # construction of force function
        if f is None:
            check_f = True
            f = self.construct_f(full_space_v, v_test)
        else:
            check_f = False

        # construction of desired state
        if v_d is None:
            check_v_d = True
            v_d = self.construct_v_d(full_space_v, v_test)
        else:
            check_v_d = False

        v_old = self._v
        v_n_help = Function(self.space_v)

        M_v = inner(v_trial, v_test) * dx

        # construction of the blocks of the system
        block_00 = {}
        block_01 = {}
        block_10 = {}
        block_11 = {}

        if not self._CN:
            for j in range(n_t):
                block_11[(0, j)] = None

        t = t_0 - tau
        for i in range(n_t - 1):
            t += tau
            v_n_help.assign(v_old.sub(i))

            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_n_help, Constant(t))
            D_zeta_i = adjoint(D_v_i)

            if not self._CN:
                for j in range(n_t):
                    if j == i - 1:
                        block_00[(i, j)] = None
                        block_01[(i, j)] = None
                        block_10[(i, j)] = -M_v
                        block_11[(i + 1, j)] = None
                    elif j == i:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_01[(i, j)] = const_tau * D_zeta_i + M_v
                        block_10[(i, j)] = const_tau * D_v_i + M_v
                        block_11[(i + 1, j)] = None
                    elif j == i + 1:
                        block_00[(i, j)] = None
                        block_01[(i, j)] = -M_v
                        block_10[(i, j)] = None
                        block_11[(i + 1, j)] = - const_tau_beta * self._M_zeta
                    else:
                        block_00[(i, j)] = None
                        block_01[(i, j)] = None
                        block_10[(i, j)] = None
                        block_11[(i + 1, j)] = None
            else:
                v_n_help_plus = Function(self.space_v)
                v_n_help_plus.assign(v_old.sub(i + 1))

                D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                v_n_help_plus,
                                                Constant(t + tau))
                D_zeta_i_plus = adjoint(D_v_i_plus)

                for j in range(n_t - 1):
                    if j == i - 1:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_01[(i, j)] = None
                        block_10[(i, j)] = const_tau * D_v_i - M_v
                        block_11[(i, j)] = None
                    elif j == i:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_01[(i, j)] = const_tau * D_zeta_i + M_v
                        block_10[(i, j)] = const_tau * D_v_i_plus + M_v
                        block_11[(i, j)] = - const_tau_beta * self._M_zeta
                    elif j == i + 1:
                        block_00[(i, j)] = None
                        block_01[(i, j)] = const_tau * D_zeta_i_plus - M_v
                        block_10[(i, j)] = None
                        block_11[(i, j)] = - const_tau_beta * self._M_zeta
                    else:
                        block_00[(i, j)] = None
                        block_01[(i, j)] = None
                        block_10[(i, j)] = None
                        block_11[(i, j)] = None

        if not self._CN:
            t = T_f
            v_n_help.assign(v_old.sub(n_t - 1))

            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_n_help, Constant(t))
            D_zeta_i = adjoint(D_v_i)

            for j in range(n_t - 2):
                block_00[(n_t - 1, j)] = None
                block_01[(n_t - 1, j)] = None
                block_10[(n_t - 1, j)] = None

            block_00[(n_t - 1, n_t - 2)] = None
            block_00[(n_t - 1, n_t - 1)] = None
            block_01[(n_t - 1, n_t - 2)] = None
            block_01[(n_t - 1, n_t - 1)] = const_tau * D_zeta_i + M_v
            block_10[(n_t - 1, n_t - 2)] = - M_v
            block_10[(n_t - 1, n_t - 1)] = const_tau * D_v_i + M_v

        # construction of right-hand side
        if not self._CN:
            b_0 = Cofunction(full_space_v.dual(), name="b_0")
            b_1 = Cofunction(full_space_v.dual(), name="b_1")
        else:
            flattened_space_v_help = tuple(self.space_v for i in range(n_t - 1))
            full_space_v_help = MixedFunctionSpace(flattened_space_v_help)

            b_0 = Cofunction(full_space_v_help.dual(), name="b_0")
            b_1 = Cofunction(full_space_v_help.dual(), name="b_1")

        if not self._CN:
            # backward Euler
            if check_v_d:
                if inhomogeneous_bcs_v:
                    v_inhom = Function(self.space_v)
                    apply_bcs(bcs_v_help[(0)], v_inhom)
                    b_0.sub(0).assign(assemble(
                        tau * v_d.sub(0)
                        - action(const_tau * self._M_v, v_inhom)))
                else:
                    b_0.sub(0).assign(tau * v_d.sub(0))

                apply_bcs(bcs_zeta, b_0.sub(0))
            else:
                b_0.sub(0).assign(v_d.sub(0))

            if check_f:
                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_0, Constant(t_0))

                if inhomogeneous_bcs_v:
                    v_inhom = Function(self.space_v)
                    apply_bcs(bcs_v_help[(0)], v_inhom)
                    b_1.sub(0).assign(assemble(
                        action(const_tau * D_v_i + M_v, v_0)
                        - action(const_tau * D_v_i + M_v, v_inhom)))
                else:
                    b_1.sub(0).assign(assemble(action(
                        const_tau * D_v_i + M_v, v_0)))

                apply_bcs(bcs_v, b_1.sub(0))
            else:
                b_1.sub(0).assign(f.sub(0))

            t = t_0
            for i in range(1, n_t - 1):
                if check_v_d:
                    if inhomogeneous_bcs_v:
                        v_inhom = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i)], v_inhom)
                        b_0.sub(i).assign(assemble(
                            tau * v_d.sub(i)
                            - action(const_tau * self._M_v, v_inhom)))
                    else:
                        b_0.sub(i).assign(tau * v_d.sub(i))

                    apply_bcs(bcs_zeta, b_0.sub(i))
                else:
                    b_0.sub(i).assign(v_d.sub(i))

                if check_f:
                    if inhomogeneous_bcs_v:
                        t += tau
                        v_n_help.assign(v_old.sub(i))

                        D_v_i = self.construct_D_v(v_trial, v_test,
                                                   v_n_help, Constant(t))

                        v_inhom0 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i - 1)], v_inhom0)
                        v_inhom_1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i)], v_inhom_1)
                        b_1.sub(i).assign(assemble(
                            tau * f.sub(i) + action(M_v, v_inhom0)
                            - action(const_tau * D_v_i + M_v, v_inhom_1)))
                    else:
                        b_1.sub(i).assign(tau * f.sub(i))

                    apply_bcs(bcs_v, b_1.sub(i))
                else:
                    b_1.sub(i).assign(f.sub(i))

            if not check_v_d:
                b_0.sub(n_t - 1).assign(v_d.sub(n_t - 1))

            if check_f:
                if inhomogeneous_bcs_v:
                    v_n_help.assign(v_old.sub(n_t - 1))

                    D_v_i = self.construct_D_v(v_trial, v_test,
                                               v_n_help, Constant(T_f))

                    v_inhom0 = Function(self.space_v)
                    apply_bcs(bcs_v_help[(n_t - 2)], v_inhom0)
                    v_inhom1 = Function(self.space_v)
                    apply_bcs(bcs_v_help[(n_t - 1)], v_inhom1)
                    b_1.sub(n_t - 1).assign(assemble(
                        tau * f.sub(n_t - 1) + action(M_v, v_inhom0)
                        - action(const_tau * D_v_i + M_v, v_inhom1)))
                else:
                    b_1.sub(n_t - 1).assign(tau * f.sub(n_t - 1))

                apply_bcs(bcs_v, b_1.sub(n_t - 1))
            else:
                b_1.sub(n_t - 1).assign(f.sub(n_t - 1))
        else:
            # trapezoidal rule
            for i in range(n_t - 1):
                if check_v_d:
                    if inhomogeneous_bcs_v:
                        v_inhom1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i + 1)], v_inhom1)

                        if i > 0:
                            v_inhom0 = Function(self.space_v)
                            apply_bcs(bcs_v_help[(i)], v_inhom0)
                            b_0.sub(i).assign(assemble(
                                0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1))
                                - action(const_tau * self._M_v, v_inhom0)
                                - action(const_tau * self._M_v, v_inhom1)))
                        else:
                            b_0.sub(i).assign(assemble(
                                0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1))
                                - action(const_tau * self._M_v, v_inhom1)))
                    else:
                        b_0.sub(i).assign(
                            0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1)))

                    apply_bcs(bcs_zeta, b_0.sub(i))
                else:
                    b_0.sub(i).assign(v_d.sub(i))

                if check_f:
                    if inhomogeneous_bcs_v:
                        t = t_0 + (i + 1) * tau
                        v_n_help.assign(v_old.sub(i + 1))
                        D_v_i = self.construct_D_v(
                            v_trial, v_test, v_n_help, Constant(t))
                        v_inhom1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i + 1)], v_inhom1)

                        if i > 0:
                            t = t_0 + i * tau
                            v_n_help.assign(v_old.sub(i))
                            D_v_j = self.construct_D_v(
                                v_trial, v_test, v_n_help, Constant(t))
                            v_inhom0 = Function(self.space_v)
                            apply_bcs(bcs_v_help[(i)], v_inhom0)

                            b_1.sub(i).assign(assemble(
                                0.5 * tau * (f.sub(i) + f.sub(i + 1))
                                - action(const_tau * D_v_j - M_v, v_inhom0)
                                - action(const_tau * D_v_i + M_v, v_inhom1)))
                        else:
                            b_1.sub(i).assign(assemble(
                                0.5 * tau * (f.sub(i) + f.sub(i + 1))
                                - action(const_tau * D_v_i + M_v, v_inhom1)))
                    else:
                        b_1.sub(i).assign(
                            0.5 * tau * (f.sub(i) + f.sub(i + 1)))

                    apply_bcs(bcs_v, b_1.sub(i))
                else:
                    b_1.sub(i).assign(f.sub(i))

            if check_v_d:
                b_0.sub(0).assign(assemble(
                    b_0.sub(0) - action(const_tau * self._M_v, v_0)))
                apply_bcs(bcs_zeta, b_0.sub(0))

            if check_f:
                D_v_i = self.construct_D_v(
                    v_trial, v_test, v_0, Constant(t_0))
                b_1.sub(0).assign(assemble(
                    b_1.sub(0) - action(const_tau * D_v_i - M_v, v_0)))
                apply_bcs(bcs_v, b_1.sub(0))

            b_0 = apply_T_1(b_0, self.space_v, n_t - 1)
            b_1 = apply_T_2(b_1, self.space_v, n_t - 1)

        # construction of the preconditioner
        if P is None:
            if self._CN:
                pc_fn = self.construct_pc(auxiliary_sp,
                                          full_space_v_help,
                                          bcs_v, bcs_zeta,
                                          block_01, block_10)
            else:
                pc_fn = self.construct_pc(auxiliary_sp,
                                          full_space_v,
                                          bcs_v, bcs_zeta,
                                          block_01, block_10,
                                          epsilon=epsilon)
        else:
            if self._CN:
                pc_fn = P(self, block_01, block_10,
                          full_space_v_help,
                          bcs_v, bcs_zeta)
            else:
                pc_fn = P(self, block_01, block_10,
                          full_space_v,
                          bcs_v, bcs_zeta)

        # solver parameters for the linear system to be solver
        if solver_parameters is None:
            solver_parameters = {"linear_solver": "gmres",
                                 "gmres_restart": 10,
                                 "maximum_iterations": 50,
                                 "relative_tolerance": 1.0e-6,
                                 "absolute_tolerance": 0.0,
                                 "monitor_convergence": True}

        # building the system
        if not self._CN:
            system = MultiBlockSystem(
                self.space_v, self.space_v,
                block_00=block_00, block_01=block_01,
                block_10=block_10, block_11=block_11,
                n_blocks_00=n_t, n_blocks_11=n_t,
                nullspace_0=full_nullspace_v,
                nullspace_1=full_nullspace_zeta,
                CN=self._CN)
        else:
            system = MultiBlockSystem(
                self.space_v, self.space_v,
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

        # solving the system
        system.solve(
            v, zeta, b_0, b_1,
            solver_parameters=solver_parameters,
            pc_fn=pc_fn)

        # updating the solutions
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
            self.set_v(v)
            self.set_zeta(zeta)

        del system, pc_fn

        if print_error:
            self.print_error(tau)

        if create_output:
            output({"v": v, "zeta": zeta})

        if plots:
            for i in range(n_t):
                plot(v.sub(i), zeta.sub(i), self._true_v.sub(i))

    @garbage_cleanup_method()
    def non_linear_solve(self, *,
                         P=None, solver_parameters=None,
                         auxiliary_sp={},
                         max_non_linear_iter=10,
                         relative_non_linear_tol=1.0e-5,
                         absolute_non_linear_tol=1.0e-8,
                         print_error_non_linear=True,
                         create_output=True,
                         plots=False):
        """Module for the solution of non-linear control problems.

        Input:
            - P                           preconditioner to apply within
                                          the Krylov method (if None,
                                          default option is employed)

            - solver_parameters           parameter to pass at the Krylov
                                          solver

            - auxiliary_sp                auxiliary parameters for setting
                                          solvers of inner blocks

            - max_non_linear_iter         maximum number of non-linear
                                          iteration

            - relative_non_linear_tol     relative non-linear tolerance

            - absolute_non_linear_tol     absolute non-linear tolerance

            - print_error_non_linear      if True, the L^2 discrepancy
                                          between the desired state and the
                                          numerical solution is printed

            - create_output               if True, output is generated

            - plots                       if True, plots of the solutions
                                          are generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

        n_t = self._n_t
        t_0 = self._time_interval[0]
        T_f = self._time_interval[1]
        tau = (T_f - t_0) / (n_t - 1.0)

        inhomogeneous_bcs_v = False
        for i, bc_i in self._bcs_v.items():
            for bc in bc_i:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

        if inhomogeneous_bcs_v:
            bcs_v = homogenize(self._bcs_v[(1)])
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v[(1)]
        bcs_zeta = bcs_v

        # full space for time integration
        flattened_space_v = tuple(self.space_v for i in range(n_t))
        full_space_v = MixedFunctionSpace(flattened_space_v)

        v_old = Function(full_space_v, name="v_old")
        zeta_old = Function(full_space_v, name="zeta_old")
        delta_v = Function(full_space_v, name="delta_v")
        delta_zeta = Function(full_space_v, name="delta_zeta")

        v_old.assign(self._v)
        zeta_old.assign(self._zeta)

        # construction of the initial condition
        if self._initial_condition is not None:
            v_0 = self._initial_condition(v_test)
        else:
            v_0 = Function(self.space_v, name="v_0")

        if self._CN:
            v_old.sub(0).assign(v_0)
        zeta_old.sub(n_t - 1).assign(Constant(0.0))

        # construction of the force function
        f = self.construct_f(full_space_v, v_test)

        # construction of the desired state
        v_d = self.construct_v_d(full_space_v, v_test)

        M_v = inner(v_trial, v_test) * dx

        if self._CN:
            flattened_space_v_help = tuple(self.space_v for i in range(n_t - 1))
            full_space_v_help = MixedFunctionSpace(flattened_space_v_help)

        # building the non-linear residual
        if self._CN:
            rhs_0, rhs_1 = self.non_linear_res_eval(
                full_space_v_help, v_old, zeta_old, v_0,
                v_d, f, M_v, bcs_v, bcs_zeta)
        else:
            rhs_0, rhs_1 = self.non_linear_res_eval(
                full_space_v, v_old, zeta_old, v_0,
                v_d, f, M_v, bcs_v, bcs_zeta)

        if not self._CN:
            rhs = Cofunction((full_space_v * full_space_v).dual(), name="rhs")
            for i in range(n_t):
                rhs.sub(i).assign(rhs_0.sub(i))
                rhs.sub(n_t + i).assign(rhs_1.sub(i))
        else:
            rhs = Cofunction((full_space_v_help * full_space_v_help).dual(), name="rhs")
            for i in range(n_t - 1):
                rhs.sub(i).assign(rhs_0.sub(i))
                rhs.sub(n_t - 1 + i).assign(rhs_1.sub(i))

        # norm of the initial non-linear residual
        norm_0 = vnorm(rhs)
        norm_k = norm_0

        k = 0

        print(f'Initial non-linear residual: {norm_0:.16e}')

        while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):
            # solving for the linearized system
            self.linear_solve(
                P=P, solver_parameters=solver_parameters,
                auxiliary_sp=auxiliary_sp, v_d=rhs_0, f=rhs_1,
                print_error=False, create_output=False, plots=False)

            delta_v.assign(self._v)
            delta_zeta.assign(self._zeta)

            # updating the solutions
            v_old += delta_v
            if inhomogeneous_bcs_v:
                for i in range(n_t):
                    apply_bcs(bcs_v_help[(i)], v_old.sub(i))

            zeta_old += delta_zeta
            for i in range(n_t):
                apply_bcs(bcs_zeta, zeta_old.sub(i))

            self.set_v(v_old)
            self.set_zeta(zeta_old)

            # evaluating non-linear residual
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

            # norm of non-linear residual
            norm_k = vnorm(rhs)

            k += 1

            print(f'Non-linear solver: '
                  f'iteration {k:d}, '
                  f'non-linear residual norm {norm_k:.16e}')

            if k + 1 > max_non_linear_iter:
                break

        if print_error_non_linear:
            if (norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol):
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
                print(f'Number of non-linear iterations: {k:d}')
            else:
                print('The non-linear iteration did not converge')
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
            self.print_error(tau)

        if create_output:
            output({"v": self._v, "zeta": self._zeta})

        if plots:
            for i in range(n_t):
                plot(self._v.sub(i), self._zeta.sub(i), self._true_v.sub(i))

    @garbage_cleanup_method()
    def incompressible_linear_solve(self, nullspace_p, *, space_p=None,
                                    P=None, solver_parameters=None,
                                    auxiliary_sp={},
                                    v_d=None, f=None,
                                    div_v=None, div_zeta=None,
                                    print_error=True,
                                    create_output=True, plots=False):
        """Module for the solution of linear incompressible control problems.

        Input:
            - nullspace_p              nullspace of the corresponding
                                       forward stationary incompressible
                                       problem

            - space_p                  pressure space, if not passed to
                                       the constructor

            - P                        preconditioner to apply within
                                       the Krylov method (if None, default
                                       option is employed)

            - solver_parameters        parameter to pass at the Krylov
                                       solver

            - auxiliary_sp             auxiliary parameters for setting
                                       solvers of inner blocks

            - v_d                      when solving non-linear problems,
                                       v_d is the non-linear residual
                                       (adjoint equation)

            - f                        when solving non-linear problems, f
                                       is the non-linear residual (state
                                       equation)

            - div_v                    when solving non-linear problems,
                                       div_v is the non-linear residual
                                       (incompressibility constraint on
                                       state variable)

            - div_zeta                 when solving non-linear problems,
                                       div_zeta is the non-linear residual
                                       (incompressibility constraint on
                                       adjoint variable)

            - print_error              if True, the L^2 discrepancy between
                                       the desired state and the numerical
                                       solution is printed

            - create_output            if True, output is generated

            - plots                    if True, plots of the solutions
                                       are generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)

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
        if not self._CN:
            const_tau = Constant(tau)
            const_tau_beta = Constant(tau / self.beta)
        else:
            const_tau = Constant(0.5 * tau)
            const_tau_help = Constant(tau)
            const_tau_beta = Constant(0.5 * tau / self.beta)

        if not self._CN:
            epsilon = Constant(1.0e-3)

        inhomogeneous_bcs_v = False
        for i, bc_i in self._bcs_v.items():
            for bc in bc_i:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

        if inhomogeneous_bcs_v:
            bcs_v = homogenize(self._bcs_v[(1)])
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v[(1)]
        bcs_zeta = bcs_v

        # construction of nullspaces
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

        # construction of full space for time integration
        flattened_space_v = tuple(self.space_v for i in range(n_t))
        full_space_v = MixedFunctionSpace(flattened_space_v)
        if not self._CN:
            full_flattened_space_v = flattened_space_v + flattened_space_v
            space_0 = MixedFunctionSpace(full_flattened_space_v)
        else:
            flattened_space_v_help = tuple(
                self.space_v for i in range(n_t - 1))
            full_space_v_help = MixedFunctionSpace(flattened_space_v_help)
            space_0 = MixedFunctionSpace(
                flattened_space_v_help + flattened_space_v_help)

        if not self._CN:
            flattened_space_p = tuple(space_p for i in range(n_t))
        else:
            flattened_space_p = tuple(space_p for i in range(n_t - 1))
        full_space_p = MixedFunctionSpace(flattened_space_p)

        full_flattened_space_p = flattened_space_p + flattened_space_p
        space_1 = MixedFunctionSpace(full_flattened_space_p)

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

        # construction of initial condition
        if self._initial_condition is not None:
            v_0 = self._initial_condition(v_test)
        else:
            v_0 = Function(self.space_v, name="v_0")

        # construction of force function
        if f is None:
            check_f = True
            f = self.construct_f(full_space_v, v_test)
        else:
            check_f = False

        # construction of desired state
        if v_d is None:
            check_v_d = True
            v_d = self.construct_v_d(full_space_v, v_test)
        else:
            check_v_d = False

        v_old = self._v
        v_n_help = Function(self.space_v)

        M_v = inner(v_trial, v_test) * dx
        B = - inner(div(v_trial), p_test) * dx
        B_T = - inner(p_trial, div(v_test)) * dx

        # construction of the blocks of the whole system and
        # of the blocks for the inner system
        block_00 = {}
        block_01 = {}
        block_10 = {}
        block_11 = {}

        block_00_int = {}
        block_01_int = {}
        block_10_int = {}
        block_11_int = {}

        if P is None:
            block_00_int_p = {}
            block_01_int_p = {}
            block_10_int_p = {}
            block_11_int_p = {}

            if self._CN:
                if self._M_p is not None:
                    block_00_p = const_tau * self._M_p
                else:
                    block_00_p = const_tau * inner(p_trial, p_test) * dx
                if self._M_mu is not None:
                    block_11_p = - const_tau_beta * self._M_mu
                else:
                    block_11_p = - const_tau_beta * inner(p_trial, p_test) * dx
            else:
                if self._M_p is not None:
                    block_00_p = const_tau * self._M_p
                else:
                    block_00_p = const_tau * inner(p_trial, p_test) * dx
                if self._M_mu is not None:
                    block_11_p = - const_tau_beta * self._M_mu
                else:
                    block_11_p = - const_tau_beta * inner(p_trial, p_test) * dx

            K_p = inner(grad(p_trial), grad(p_test)) * dx
            M_p = inner(p_trial, p_test) * dx

        if not self._CN:
            for i in range(2 * n_t):
                for j in range(2 * n_t):
                    block_11[(i, j)] = None
                    if j == i:
                        block_01[(i, j)] = const_tau * B_T
                        block_10[(i, j)] = const_tau * B
                    else:
                        block_01[(i, j)] = None
                        block_10[(i, j)] = None
        else:
            for i in range(2 * n_t - 2):
                for j in range(2 * n_t - 2):
                    block_11[(i, j)] = None
                    if j == i:
                        block_01[(i, j)] = const_tau_help * B_T
                        block_10[(i, j)] = const_tau_help * B
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

            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_n_help, Constant(t))
            D_zeta_i = adjoint(D_v_i)

            if P is None:
                D_p_i = self.construct_D_v(p_trial, p_test,
                                           v_n_help, Constant(t),
                                           non_linear_res=True)
                D_mu_i = adjoint(D_p_i)

            if not self._CN:
                for j in range(n_t):
                    if j == i - 1:
                        block_00[(i, j)] = None
                        block_00[(i, n_t + j)] = None
                        block_00[(n_t + i, j)] = -M_v
                        block_00[(n_t + i + 1, n_t + j)] = None

                        block_00_int[(i, j)] = None
                        block_01_int[(i, j)] = None
                        block_10_int[(i, j)] = -M_v
                        block_11_int[(i + 1, j)] = None

                        if P is None:
                            block_10_int_p[(i, j)] = -M_p
                    elif j == i:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_00[(i, n_t + j)] = const_tau * D_zeta_i + M_v
                        block_00[(n_t + i, j)] = const_tau * D_v_i + M_v
                        block_00[(n_t + i + 1, n_t + j)] = None

                        block_00_int[(i, j)] = const_tau * self._M_v
                        block_01_int[(i, j)] = const_tau * D_zeta_i + M_v
                        block_10_int[(i, j)] = const_tau * D_v_i + M_v
                        block_11_int[(i + 1, j)] = None

                        if P is None:
                            block_00_int_p[(i, j)] = block_00_p
                            block_01_int_p[(i, j)] = const_tau * D_mu_i + M_p
                            block_10_int_p[(i, j)] = const_tau * D_p_i + M_p
                    elif j == i + 1:
                        block_00[(i, j)] = None
                        block_00[(i, n_t + j)] = -M_v
                        block_00[(n_t + i, j)] = None
                        block_00[(n_t + i + 1, n_t + j)] = - const_tau_beta * self._M_zeta

                        block_00_int[(i, j)] = None
                        block_01_int[(i, j)] = -M_v
                        block_10_int[(i, j)] = None
                        block_11_int[(i + 1, j)] = - const_tau_beta * self._M_zeta

                        if P is None:
                            block_01_int_p[(i, j)] = -M_p
                            block_11_int_p[(i + 1, j)] = block_11_p
                    else:
                        block_00[(i, j)] = None
                        block_00[(i, n_t + j)] = None
                        block_00[(n_t + i, j)] = None
                        block_00[(n_t + i + 1, n_t + j)] = None

                        block_00_int[(i, j)] = None
                        block_01_int[(i, j)] = None
                        block_10_int[(i, j)] = None
                        block_11_int[(i + 1, j)] = None
            else:
                v_n_help_plus = Function(self.space_v)
                v_n_help_plus.assign(v_old.sub(i + 1))

                D_v_i_plus = self.construct_D_v(v_trial, v_test,
                                                v_n_help_plus,
                                                Constant(t + tau))
                D_zeta_i_plus = adjoint(D_v_i_plus)

                if P is None:
                    D_p_i_plus = self.construct_D_v(
                        p_trial, p_test, v_n_help_plus,
                        Constant(t + tau), non_linear_res=True)
                    D_mu_i_plus = adjoint(D_p_i_plus)

                for j in range(n_t - 1):
                    if j == i - 1:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_00[(i, n_t + j - 1)] = None
                        block_00[(n_t + i - 1, j)] = const_tau * D_v_i - M_v
                        block_00[(n_t + i - 1, n_t + j - 1)] = None

                        block_00_int[(i, j)] = const_tau * self._M_v
                        block_01_int[(i, j)] = None
                        block_10_int[(i, j)] = const_tau * D_v_i - M_v
                        block_11_int[(i, j)] = None

                        if P is None:
                            block_00_int_p[(i, j)] = block_00_p
                            block_10_int_p[(i, j)] = const_tau * D_p_i - M_p
                    elif j == i:
                        block_00[(i, j)] = const_tau * self._M_v
                        block_00[(i, n_t + j - 1)] = const_tau * D_zeta_i + M_v
                        block_00[(n_t + i - 1, j)] = const_tau * D_v_i_plus + M_v
                        block_00[(n_t + i - 1, n_t + j - 1)] = - const_tau_beta * self._M_zeta

                        block_00_int[(i, j)] = const_tau * self._M_v
                        block_01_int[(i, j)] = const_tau * D_zeta_i + M_v
                        block_10_int[(i, j)] = const_tau * D_v_i_plus + M_v
                        block_11_int[(i, j)] = - const_tau_beta * self._M_zeta

                        if P is None:
                            block_00_int_p[(i, j)] = block_00_p
                            block_01_int_p[(i, j)] = const_tau * D_mu_i + M_p
                            block_10_int_p[(i, j)] = const_tau * D_p_i_plus + M_p
                            block_11_int_p[(i, j)] = block_11_p
                    elif j == i + 1:
                        block_00[(i, j)] = None
                        block_00[(i, n_t + j - 1)] = const_tau * D_zeta_i_plus - M_v
                        block_00[(n_t + i - 1, j)] = None
                        block_00[(n_t + i - 1, n_t + j - 1)] = - const_tau_beta * self._M_zeta

                        block_00_int[(i, j)] = None
                        block_01_int[(i, j)] = const_tau * D_zeta_i_plus - M_v
                        block_10_int[(i, j)] = None
                        block_11_int[(i, j)] = - const_tau_beta * self._M_zeta

                        if P is None:
                            block_01_int_p[(i, j)] = const_tau * D_mu_i_plus - M_p
                            block_11_int_p[(i, j)] = block_11_p
                    else:
                        block_00[(i, j)] = None
                        block_00[(i, n_t + j - 1)] = None
                        block_00[(n_t + i - 1, j)] = None
                        block_00[(n_t + i - 1, n_t + j - 1)] = None

                        block_00_int[(i, j)] = None
                        block_01_int[(i, j)] = None
                        block_10_int[(i, j)] = None
                        block_11_int[(i, j)] = None

        if not self._CN:
            t = T_f
            v_n_help.assign(v_old.sub(n_t - 1))

            D_v_i = self.construct_D_v(v_trial, v_test,
                                       v_n_help, Constant(t))
            D_zeta_i = adjoint(D_v_i)

            if P is None:
                D_p_i = self.construct_D_v(p_trial, p_test,
                                           v_n_help, Constant(t),
                                           non_linear_res=True)
                D_mu_i = adjoint(D_p_i)

            for j in range(n_t - 2):
                block_00[(n_t - 1, j)] = None
                block_00[(n_t - 1, n_t + j)] = None
                block_00[(2 * n_t - 1, j)] = None

                block_00_int[(n_t - 1, j)] = None
                block_01_int[(n_t - 1, j)] = None
                block_10_int[(n_t - 1, j)] = None

            block_00[(n_t - 1, n_t - 2)] = None
            block_00[(n_t - 1, n_t - 1)] = None
            block_00[(n_t - 1, 2 * n_t - 2)] = None
            block_00[(n_t - 1, 2 * n_t - 1)] = const_tau * D_zeta_i + M_v
            block_00[(2 * n_t - 1, n_t - 2)] = - M_v
            block_00[(2 * n_t - 1, n_t - 1)] = const_tau * D_v_i + M_v

            block_00_int[(n_t - 1, n_t - 2)] = None
            block_00_int[(n_t - 1, n_t - 1)] = None
            block_01_int[(n_t - 1, n_t - 2)] = None
            block_01_int[(n_t - 1, n_t - 1)] = const_tau * D_zeta_i + M_v
            block_10_int[(n_t - 1, n_t - 2)] = - M_v
            block_10_int[(n_t - 1, n_t - 1)] = const_tau * D_v_i + M_v

            if P is None:
                block_01_int_p[(n_t - 1, n_t - 1)] = const_tau * D_mu_i + M_p
                block_10_int_p[(n_t - 1, n_t - 2)] = - M_p
                block_10_int_p[(n_t - 1, n_t - 1)] = const_tau * D_p_i + M_p

        # construction of the right-hand side
        if not self._CN:
            # backward Euler
            if check_v_d:
                if inhomogeneous_bcs_v:
                    v_inhom = Function(self.space_v)
                    apply_bcs(bcs_v_help[(0)], v_inhom)
                    b_0_0.sub(0).assign(assemble(
                        tau * v_d.sub(0) - action(const_tau * self._M_v, v_inhom)))
                else:
                    b_0_0.sub(0).assign(tau * v_d.sub(0))
                apply_bcs(bcs_v, b_0_0.sub(0))
            else:
                b_0_0.sub(0).assign(v_d.sub(0))

            if check_f:
                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_0, Constant(t_0))

                if inhomogeneous_bcs_v:
                    v_inhom = Function(self.space_v)
                    apply_bcs(bcs_v_help[(0)], v_inhom)
                    b_0_1.sub(0).assign(assemble(
                        action(const_tau * D_v_i + M_v, v_0)
                        - action(const_tau * D_v_i + M_v, v_inhom)))
                else:
                    b_0_1.sub(0).assign(assemble(
                        action(const_tau * D_v_i + M_v, v_0)))

                apply_bcs(bcs_zeta, b_0_1.sub(0))
            else:
                b_0_1.sub(0).assign(f.sub(0))

            t = t_0
            for i in range(1, n_t - 1):
                if check_v_d:
                    if inhomogeneous_bcs_v:
                        v_inhom = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i)], v_inhom)
                        b_0_0.sub(i).assign(assemble(
                            tau * v_d.sub(i)
                            - action(const_tau * self._M_v, v_inhom)))
                    else:
                        b_0_0.sub(i).assign(tau * v_d.sub(i))
                    apply_bcs(bcs_zeta, b_0_0.sub(i))
                else:
                    b_0_0.sub(i).assign(v_d.sub(i))

                if check_f:
                    if inhomogeneous_bcs_v:
                        t += tau

                        v_n_help.assign(v_old.sub(i))
                        D_v_i = self.construct_D_v(v_trial, v_test,
                                                   v_n_help, Constant(t))

                        v_inhom0 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i - 1)], v_inhom0)
                        v_inhom1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i)], v_inhom1)
                        b_0_1.sub(i).assign(assemble(
                            tau * f.sub(i) + action(M_v, v_inhom0)
                            - action(const_tau * D_v_i + M_v, v_inhom1)))
                    else:
                        b_0_1.sub(i).assign(tau * f.sub(i))
                    apply_bcs(bcs_v, b_0_1.sub(i))
                else:
                    b_0_1.sub(i).assign(f.sub(i))

            if not check_v_d:
                b_0_0.sub(n_t - 1).assign(v_d.sub(n_t - 1))

            if check_f:
                if inhomogeneous_bcs_v:
                    v_n_help.assign(v_old.sub(n_t - 1))

                    D_v_i = self.construct_D_v(v_trial, v_test,
                                               v_n_help, Constant(T_f))

                    v_inhom0 = Function(self.space_v)
                    apply_bcs(bcs_v_help[(n_t - 2)], v_inhom0)
                    v_inhom1 = Function(self.space_v)
                    apply_bcs(bcs_v_help[(n_t - 1)], v_inhom1)
                    b_0_1.sub(n_t - 1).assign(assemble(
                        tau * f.sub(n_t - 1) + action(M_v, v_inhom0)
                        - action(const_tau * D_v_i + M_v, v_inhom1)))
                else:
                    b_0_1.sub(n_t - 1).assign(
                        tau * f.sub(n_t - 1))
                apply_bcs(bcs_v, b_0_1.sub(n_t - 1))
            else:
                b_0_1.sub(n_t - 1).assign(f.sub(n_t - 1))

            if div_v is None:
                if inhomogeneous_bcs_v:
                    for i in range(n_t):
                        v_inhom = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i)], v_inhom)
                        b_1_0.sub(i).assign(assemble(
                            - action(const_tau * B, v_inhom)))
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
            # trapezoidal rule
            for i in range(n_t - 1):
                if check_v_d:
                    if inhomogeneous_bcs_v:
                        v_inhom1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i + 1)], v_inhom1)
                        if i > 0:
                            v_inhom0 = Function(self.space_v)
                            apply_bcs(bcs_v_help[(i)], v_inhom0)
                            b_0_0.sub(i).assign(assemble(
                                0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1))
                                - action(const_tau * self._M_v, v_inhom0)
                                - action(const_tau * self._M_v, v_inhom1)))
                        else:
                            b_0_0.sub(i).assign(assemble(
                                0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1))
                                - action(const_tau * self._M_v, v_inhom1)))
                    else:
                        b_0_0.sub(i).assign(
                            0.5 * tau * (v_d.sub(i) + v_d.sub(i + 1)))
                    apply_bcs(bcs_zeta, b_0_0.sub(i))
                else:
                    b_0_0.sub(i).assign(v_d.sub(i))

                if check_f:
                    if inhomogeneous_bcs_v:
                        t = t_0 + (i + 1) * tau
                        v_n_help.assign(v_old.sub(i + 1))
                        D_v_j = self.construct_D_v(
                            v_trial, v_test, v_n_help, Constant(t))

                        v_inhom1 = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i + 1)], v_inhom1)
                        if i > 0:
                            t = t_0 + i * tau
                            v_n_help.assign(v_old.sub(i))
                            D_v_i = self.construct_D_v(
                                v_trial, v_test, v_n_help, Constant(t))

                            v_inhom0 = Function(self.space_v)
                            apply_bcs(bcs_v_help[(i)], v_inhom0)
                            b_0_1.sub(i).assign(assemble(
                                0.5 * tau * (f.sub(i) + f.sub(i + 1))
                                - action(const_tau * D_v_i - M_v, v_inhom0)
                                - action(const_tau * D_v_j + M_v, v_inhom1)))
                        else:
                            b_0_1.sub(i).assign(assemble(
                                0.5 * tau * (f.sub(i) + f.sub(i + 1))
                                - action(const_tau * D_v_j + M_v, v_inhom1)))
                    else:
                        b_0_1.sub(i).assign(
                            0.5 * tau * (f.sub(i) + f.sub(i + 1)))
                    apply_bcs(bcs_v, b_0_1.sub(i))
                else:
                    b_0_1.sub(i).assign(f.sub(i))

            if check_v_d:
                b_0_0.sub(0).assign(assemble(
                    b_0_0.sub(0)
                    - action(const_tau * self._M_v, v_0)))
                apply_bcs(bcs_zeta, b_0_0.sub(0))

            if check_f:
                D_v_i = self.construct_D_v(v_trial, v_test,
                                           v_0, Constant(t_0))

                b_0_1.sub(0).assign(assemble(
                    b_0_1.sub(0)
                    - action(const_tau * D_v_i - M_v, v_0)))
                apply_bcs(bcs_v, b_0_1.sub(0))

            if div_v is None:
                if inhomogeneous_bcs_v:
                    for i in range(n_t - 1):
                        v_inhom = Function(self.space_v)
                        apply_bcs(bcs_v_help[(i + 1)], v_inhom)
                        b_1_0.sub(i).assign(assemble(
                            - action(const_tau_help * B, v_inhom)))
            else:
                for i in range(n_t - 1):
                    b_1_0.sub(i).assign(div_v.sub(i))

            if div_zeta is not None:
                for i in range(n_t - 1):
                    b_1_1.sub(i).assign(div_zeta.sub(i))

            b_0_0 = apply_T_1(b_0_0, self.space_v, n_t - 1)
            b_0_1 = apply_T_2(b_0_1, self.space_v, n_t - 1)

            b_1_0 = apply_T_2(b_1_0, space_p, n_t - 1)
            b_1_1 = apply_T_1(b_1_1, space_p, n_t - 1)

            for i in range(n_t - 1):
                index = n_t - 1 + i

                b_0.sub(i).assign(b_0_0.sub(i))
                b_0.sub(index).assign(b_0_1.sub(i))

                b_1.sub(i).assign(b_1_0.sub(i))
                b_1.sub(index).assign(b_1_1.sub(i))

        # construction of the system
        if not self._CN:
            system = MultiBlockSystem(
                self.space_v, space_p,
                block_00=block_00, block_01=block_01,
                block_10=block_10, block_11=block_11,
                n_blocks_00=2 * n_t, n_blocks_11=2 * n_t,
                nullspace_0=full_nullspace_0, nullspace_1=full_nullspace_1,
                CN=self._CN)
        else:
            system = MultiBlockSystem(
                self.space_v, space_p,
                block_00=block_00, block_01=block_01,
                block_10=block_10, block_11=block_11,
                n_blocks_00=2 * n_t - 2, n_blocks_11=2 * n_t - 2,
                sub_n_blocks_00_0=n_t - 1, sub_n_blocks_11_0=n_t - 1,
                nullspace_0=full_nullspace_0, nullspace_1=full_nullspace_1,
                CN=self._CN)

        # solver parameters for the whole system
        if solver_parameters is None:
            solver_parameters = {"linear_solver": "fgmres",
                                 "fgmres_restart": 10,
                                 "maximum_iterations": 100,
                                 "relative_tolerance": 1.0e-6,
                                 "absolute_tolerance": 0.0,
                                 "monitor_convergence": True}

        # construction of the preconditioner
        if P is None:
            # sp for the pressure stiffness matrix
            if "sp_K_p" in auxiliary_sp:
                sp_K_p = auxiliary_sp["sp_K_p"]
            else:
                sp_K_p = {"ksp_type": "preonly",
                          "pc_type": "hypre",
                          "pc_hypre_type": "boomeramg",
                          "ksp_max_it": 1,
                          "pc_hypre_boomeramg_max_iter": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

            # sp for the pressure mass matrix
            if "sp_M_p" in auxiliary_sp:
                sp_M_p = auxiliary_sp["sp_M_p"]
            else:
                sp_M_p = {"ksp_type": "preonly",
                          "pc_type": "jacobi",
                          "ksp_max_it": 1,
                          "ksp_atol": 0.0,
                          "ksp_rtol": 0.0}

            # solver for the pressure stiffness matrix
            solver_K_p = LinearSolver(
                assemble(K_p),
                solver_parameters=sp_K_p)
            solver_K_p.ksp.addConvergenceTest(converged, prepend=True)

            # solver for the pressure mass matrix
            solver_M_p = LinearSolver(
                assemble(M_p),
                solver_parameters=sp_M_p)
            solver_M_p.ksp.addConvergenceTest(converged, prepend=True)

            # parameters for the inner system
            if "sp_inner" in auxiliary_sp:
                inner_solver_parameters = auxiliary_sp["sp_inner"]
            else:
                inner_solver_parameters = {
                    "preconditioner": True,
                    "linear_solver": "gmres",
                    "maximum_iterations": 5,
                    "relative_tolerance": 0.0,
                    "absolute_tolerance": 0.0,
                    "monitor_convergence": False}

            # definition of preconditioner
            if self._CN:
                # inner solver for the trapezoidal rule
                self._inner_system = MultiBlockSystem(
                    self.space_v, self.space_v,
                    block_00=block_00_int, block_01=block_01_int,
                    block_10=block_10_int, block_11=block_11_int,
                    n_blocks_00=n_t - 1, n_blocks_11=n_t - 1,
                    nullspace_0=full_nullspace_v,
                    nullspace_1=full_nullspace_zeta,
                    CN=self._CN)

                self._inner_pc_fn = self.construct_pc(
                    auxiliary_sp, full_space_v_help,
                    bcs_v, bcs_zeta, block_01_int, block_10_int)

                # preconditioner for the trapezoidal rule
                @garbage_cleanup(self.comm)
                def pc_fn(u_0, u_1, b_0, b_1):
                    b_0_help = Cofunction(full_space_v_help.dual())
                    b_1_help = Cofunction(full_space_v_help.dual())

                    for i in range(n_t - 1):
                        b_0_help.sub(i).assign(b_0.sub(i))
                        index = n_t - 1 + i
                        b_1_help.sub(i).assign(b_0.sub(index))

                    v_help = Function(full_space_v_help)
                    zeta_help = Function(full_space_v_help)

                    # solving for the (1,1)-block
                    _ = self._inner_system.solve(
                        v_help, zeta_help, b_0_help, b_1_help,
                        solver_parameters=inner_solver_parameters,
                        pc_fn=self._inner_pc_fn)

                    for i in range(n_t - 1):
                        u_0.sub(i).assign(v_help.sub(i))
                        index = n_t - 1 + i
                        u_0.sub(index).assign(zeta_help.sub(i))

                    # u_1 = - b_1 + block_10 * u_0
                    b_0_help = Cofunction(full_space_p.dual())
                    b_1_help = Cofunction(full_space_p.dual())
                    v_help = Function(self.space_v)
                    for i in range(n_t - 1):
                        v_help.assign(u_0.sub(i))
                        b_0_help.sub(i).assign(assemble(action(B, v_help)))
                        with b_0_help.sub(i).dat.vec as b_v:
                            b_v.scale(tau)

                        v_help.assign(u_0.sub(n_t - 1 + i))
                        b_1_help.sub(i).assign(assemble(action(B, v_help)))
                        with b_1_help.sub(i).dat.vec as b_v:
                            b_v.scale(tau)

                    b_0_help = apply_T_2(b_0_help, space_p, n_t - 1)
                    b_1_help = apply_T_1(b_1_help, space_p, n_t - 1)

                    for i in range(n_t - 1):
                        b_0_help.sub(i).assign(assemble(
                            b_0_help.sub(i) - b_1.sub(i)))
                        index = n_t - 1 + i
                        b_1_help.sub(i).assign(assemble(
                            b_1_help.sub(i) - b_1.sub(index)))

                    # solving for the Schur complement approximation (apply
                    # block-pressure convection--diffusion preconditioner)
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
                        u_1.sub(i).zero()
                        solver_K_p.solve(u_1.sub(i),
                                         p_help.copy(deepcopy=True))

                        index = n_t - 1 + i
                        p_help.assign(b_1_help.sub(i))
                        u_1.sub(index).zero()
                        solver_K_p.solve(u_1.sub(index),
                                         p_help.copy(deepcopy=True))

                    b_0_help = Cofunction(full_space_p.dual())
                    b_1_help = Cofunction(full_space_p.dual())

                    p_help = Function(space_p)
                    mu_help = Function(space_p)

                    for (i, j), block_ij_p in block_00_int_p.items():
                        if block_ij_p is not None:
                            p_help.assign(u_1.sub(j))
                            b_0_help.sub(i).assign(assemble(
                                b_0_help.sub(i) + action(block_ij_p, p_help)))

                    for (i, j), block_ij_p in block_01_int_p.items():
                        if block_ij_p is not None:
                            mu_help.assign(u_1.sub(n_t - 1 + j))
                            b_0_help.sub(i).assign(assemble(
                                b_0_help.sub(i) + action(block_ij_p, mu_help)))

                    for (i, j), block_ij_p in block_10_int_p.items():
                        if block_ij_p is not None:
                            p_help.assign(u_1.sub(j))
                            b_1_help.sub(i).assign(assemble(
                                b_1_help.sub(i) + action(block_ij_p, p_help)))

                    for (i, j), block_ij_p in block_11_int_p.items():
                        if block_ij_p is not None:
                            mu_help.assign(u_1.sub(n_t - 1 + j))
                            b_1_help.sub(i).assign(assemble(
                                b_1_help.sub(i) + action(block_ij_p, mu_help)))

                    p_help = Cofunction(space_p.dual())
                    for i in range(n_t - 1):
                        p_help.assign(b_0_help.sub(i))
                        u_1.sub(i).zero()
                        solver_M_p.solve(u_1.sub(i),
                                         p_help.copy(deepcopy=True))

                        index = n_t - 1 + i
                        p_help.assign(b_1_help.sub(i))
                        u_1.sub(index).zero()
                        solver_M_p.solve(u_1.sub(index),
                                         p_help.copy(deepcopy=True))
            else:
                # inner solver for backward Euler
                self._inner_system = MultiBlockSystem(
                    self.space_v, self.space_v,
                    block_00=block_00_int, block_01=block_01_int,
                    block_10=block_10_int, block_11=block_11_int,
                    n_blocks_00=n_t, n_blocks_11=n_t,
                    nullspace_0=full_nullspace_v,
                    nullspace_1=full_nullspace_zeta,
                    CN=self._CN)

                self._inner_pc_fn = self.construct_pc(
                    auxiliary_sp, full_space_v,
                    bcs_v, bcs_zeta, block_01_int, block_10_int,
                    epsilon=epsilon)

                # preconditioner for bacward Euler
                @garbage_cleanup(self.comm)
                def pc_fn(u_0, u_1, b_0, b_1):
                    b_0_help = Cofunction(full_space_v.dual())
                    b_1_help = Cofunction(full_space_v.dual())

                    for i in range(n_t):
                        b_0_help.sub(i).assign(b_0.sub(i))
                        index = n_t + i
                        b_1_help.sub(i).assign(b_0.sub(index))

                    v_help = Function(full_space_v)
                    zeta_help = Function(full_space_v)

                    # solving for the (1,1)-block
                    _ = self._inner_system.solve(
                        v_help, zeta_help, b_0_help, b_1_help,
                        solver_parameters=inner_solver_parameters,
                        pc_fn=self._inner_pc_fn)

                    for i in range(n_t):
                        u_0.sub(i).assign(v_help.sub(i))
                        index = n_t + i
                        u_0.sub(index).assign(zeta_help.sub(i))

                    # u_1 = - b_1 + block_10 * u_0
                    b_0_help = Cofunction(full_space_p.dual())
                    b_1_help = Cofunction(full_space_p.dual())
                    v_help = Function(self.space_v)
                    for i in range(n_t):
                        v_help.assign(u_0.sub(i))
                        b_0_help.sub(i).assign(assemble(action(B, v_help)))
                        with b_0_help.sub(i).dat.vec as b_v:
                            b_v.scale(tau)

                        v_help.assign(u_0.sub(n_t + i))
                        b_1_help.sub(i).assign(assemble(action(B, v_help)))
                        with b_1_help.sub(i).dat.vec as b_v:
                            b_v.scale(tau)

                    for i in range(n_t):
                        b_0_help.sub(i).assign(assemble(
                            b_0_help.sub(i) - b_1.sub(i)))

                        index = n_t + i
                        b_1_help.sub(i).assign(assemble(
                            b_1_help.sub(i) - b_1.sub(index)))

                    # solving for the Schur complement approximation (apply
                    # block-pressure convection--diffusion preconditioner)
                    for i in range(n_t):
                        with b_0_help.sub(i).dat.vec as b_v:
                            b_v.scale(1.0 / (tau**2))
                        with b_1_help.sub(i).dat.vec as b_v:
                            b_v.scale(1.0 / (tau**2))

                    p_help = Cofunction(space_p.dual())
                    for i in range(n_t):
                        p_help.assign(b_0_help.sub(i))
                        u_1.sub(i).zero()
                        solver_K_p.solve(u_1.sub(i),
                                         p_help.copy(deepcopy=True))

                        index = n_t + i
                        p_help.assign(b_1_help.sub(i))
                        u_1.sub(index).zero()
                        solver_K_p.solve(u_1.sub(index),
                                         p_help.copy(deepcopy=True))

                    b_0_help = Cofunction(full_space_p.dual())
                    b_1_help = Cofunction(full_space_p.dual())

                    p_help = Function(space_p)
                    mu_help = Function(space_p)

                    for (i, j), block_ij_p in block_00_int_p.items():
                        if block_ij_p is not None:
                            p_help.assign(u_1.sub(j))
                            b_0_help.sub(i).assign(assemble(
                                b_0_help.sub(i) + action(block_ij_p, p_help)))

                    for (i, j), block_ij_p in block_01_int_p.items():
                        if block_ij_p is not None:
                            mu_help.assign(u_1.sub(n_t + j))
                            b_0_help.sub(i).assign(assemble(
                                b_0_help.sub(i) + action(block_ij_p, mu_help)))

                    for (i, j), block_ij_p in block_10_int_p.items():
                        if block_ij_p is not None:
                            p_help.assign(u_1.sub(j))
                            b_1_help.sub(i).assign(assemble(
                                b_1_help.sub(i) + action(block_ij_p, p_help)))

                    for (i, j), block_ij_p in block_11_int_p.items():
                        if block_ij_p is not None:
                            mu_help.assign(u_1.sub(n_t + j))
                            b_1_help.sub(i).assign(assemble(
                                b_1_help.sub(i) + action(block_ij_p, mu_help)))

                    p_help = Cofunction(space_p.dual())
                    for i in range(n_t):
                        p_help.assign(b_0_help.sub(i))
                        u_1.sub(i).zero()
                        solver_M_p.solve(u_1.sub(i),
                                         p_help.copy(deepcopy=True))

                        index = n_t + i
                        p_help.assign(b_1_help.sub(i))
                        u_1.sub(index).zero()
                        solver_M_p.solve(u_1.sub(index),
                                         p_help.copy(deepcopy=True))
        else:
            pc_fn = P(self, block_00_int, block_01_int,
                      block_10_int, block_11_int, B,
                      full_nullspace_v, full_nullspace_zeta,
                      bcs_v, bcs_zeta)

        u_0_sol = Function(space_0)
        u_1_sol = Function(space_1)

        # solving the system
        system.solve(
            u_0_sol, u_1_sol, b_0, b_1,
            solver_parameters=solver_parameters,
            pc_fn=pc_fn)

        v = Function(full_space_v, name="v")
        zeta = Function(full_space_v, name="zeta")

        p = Function(full_space_p, name="p")
        mu = Function(full_space_p, name="mu")

        # updating the solutions
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

        self.set_v(v)
        self.set_zeta(zeta)

        self.set_p(p)
        self.set_mu(mu)

        del system, pc_fn
        if P is None:
            del self._inner_pc_fn, self._inner_system

        if print_error:
            self.print_error(tau)

        if create_output:
            output({"v": v, "zeta": zeta, "p": p, "mu": mu})

        if plots:
            for i in range(n_t - 1):
                plot(v.sub(i), p.sub(i), zeta.sub(i), mu.sub(i), self._true_v.sub(i))
            if self._CN:
                plot(v.sub(n_t - 1), zeta.sub(n_t - 1), self._true_v.sub(n_t - 1))
            else:
                plot(v.sub(n_t - 1), p.sub(n_t - 1), zeta.sub(n_t - 1), mu.sub(n_t - 1), self._true_v.sub(n_t - 1))

    @garbage_cleanup_method()
    def incompressible_non_linear_solve(self, nullspace_p, *,
                                        space_p=None, P=None,
                                        solver_parameters=None,
                                        auxiliary_sp={},
                                        max_non_linear_iter=10,
                                        relative_non_linear_tol=1.0e-5,
                                        absolute_non_linear_tol=1.0e-8,
                                        print_error_non_linear=True,
                                        create_output=True,
                                        plots=False):
        """Module for the solution of non-linear incompressible control problems.

        Input:
            - nullspace_p                nullspace of the corresponding
                                         forward stationary incompressible
                                         problem

            - space_p                    pressure space, if not passed to
                                         the constructor

            - P                          preconditioner to apply within
                                         the Krylov method (if None,
                                         default option is employed)

            - solver_parameters          parameter to pass at the Krylov
                                         solver

            - auxiliary_sp               auxiliary parameters for setting
                                         solvers of inner blocks

            - max_non_linear_iter        maximum number of non-linear
                                         iteration

            - relative_non_linear_tol    relative non-linear tolerance

            - absolute_non_linear_tol    absolute non-linear tolerance

            - print_error_non_linear     if True, the L^2 discrepancy
                                         between the desired state and the
                                         numerical solution is printed

            - create_output              if True, output is generated

            - plots                      if True, plots of the solutions
                                         are generated
        """

        v_test, v_trial = TestFunction(self.space_v), TrialFunction(self.space_v)
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
        const_tau = Constant(tau)

        inhomogeneous_bcs_v = False
        for i, bc_i in self._bcs_v.items():
            for bc in bc_i:
                if not isinstance(bc.function_arg, ufl.classes.Zero):
                    inhomogeneous_bcs_v = True

        if inhomogeneous_bcs_v:
            bcs_v = homogenize(self._bcs_v[(1)])
            bcs_v_help = self._bcs_v
        else:
            bcs_v = self._bcs_v[(1)]
        bcs_zeta = bcs_v

        # construction of the full space for time integration
        flattened_space_v = tuple(self.space_v for i in range(n_t))
        full_space_v = MixedFunctionSpace(flattened_space_v)

        if self._CN:
            flattened_space_v_help = tuple(
                self.space_v for i in range(n_t - 1))
            full_space_v_help = MixedFunctionSpace(flattened_space_v_help)

        if not self._CN:
            flattened_space_p = tuple(space_p for i in range(n_t))
        else:
            flattened_space_p = tuple(space_p for i in range(n_t - 1))
        full_space_p = MixedFunctionSpace(flattened_space_p)

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

        # construction of initial condition
        if self._initial_condition is not None:
            v_0 = self._initial_condition(v_test)
        else:
            v_0 = Function(self.space_v, name="v_0")

        if self._CN:
            v_old.sub(0).assign(v_0)
        zeta_old.sub(n_t - 1).assign(Constant(0.0))

        # construction of force function
        f = self.construct_f(full_space_v, v_test)

        # construction of desired state
        v_d = self.construct_v_d(full_space_v, v_test)

        M_v = inner(v_trial, v_test) * dx

        B = - inner(div(v_trial), p_test) * dx
        B_T = - inner(p_trial, div(v_test)) * dx

        # function used for the construction of the non-linear residual
        @garbage_cleanup(self.comm)
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

            if not self._CN:
                for i in range(n_t):
                    b_p_help = Function(space_p)
                    b_p_help.assign(mu_old.sub(i))
                    rhs_00.sub(i).assign(assemble(
                        rhs_00.sub(i) - action(const_tau * B_T, b_p_help)))
                    apply_bcs(bcs_v, rhs_00.sub(i))

                    b_p_help = Function(space_p)
                    b_p_help.assign(p_old.sub(i))
                    rhs_01.sub(i).assign(assemble(
                        rhs_01.sub(i) - action(const_tau * B_T, b_p_help)))
                    apply_bcs(bcs_zeta, rhs_01.sub(i))

                    b_help = Function(self.space_v)
                    b_help.assign(v_old.sub(i))
                    rhs_10.sub(i).assign(assemble(- action(B, b_help)))

                    b_help = Function(self.space_v)
                    b_help.assign(zeta_old.sub(i))
                    rhs_11.sub(i).assign(assemble(- action(B, b_help)))
            else:
                for i in range(n_t - 1):
                    b_p_help = Function(space_p)
                    b_p_help.assign(mu_old.sub(i))
                    rhs_00.sub(i).assign(assemble(
                        rhs_00.sub(i) - action(const_tau * B_T, b_p_help)))
                    apply_bcs(bcs_v, rhs_00.sub(i))

                    b_p_help = Function(space_p)
                    b_p_help.assign(p_old.sub(i))
                    rhs_01.sub(i).assign(assemble(
                        rhs_01.sub(i) - action(const_tau * B_T, b_p_help)))
                    apply_bcs(bcs_zeta, rhs_01.sub(i))

                    b_help = Function(self.space_v)
                    b_help.assign(v_old.sub(i + 1))
                    rhs_10.sub(i).assign(assemble(- action(B, b_help)))

                    b_help = Function(self.space_v)
                    b_help.assign(zeta_old.sub(i))
                    rhs_11.sub(i).assign(assemble(- action(B, b_help)))

            return rhs_00, rhs_01, rhs_10, rhs_11

        # evaluating the non-linear residual
        rhs_00, rhs_01, rhs_10, rhs_11 = non_linear_res_eval()

        if not self._CN:
            rhs = Cofunction((full_space_v * full_space_v * full_space_p * full_space_p).dual(), name="rhs")
            for i in range(n_t):
                rhs.sub(i).assign(rhs_00.sub(i))
                rhs.sub(n_t + i).assign(rhs_01.sub(i))
                rhs.sub(2 * n_t + i).assign(rhs_10.sub(i))
                rhs.sub(3 * n_t + i).assign(rhs_11.sub(i))
        else:
            rhs = Cofunction((full_space_v_help * full_space_v_help * full_space_p * full_space_p).dual(), name="rhs")
            for i in range(n_t - 1):
                rhs.sub(i).assign(rhs_00.sub(i))
                rhs.sub(n_t - 1 + i).assign(rhs_01.sub(i))
                rhs.sub(2 * n_t - 2 + i).assign(rhs_10.sub(i))
                rhs.sub(3 * n_t - 3 + i).assign(rhs_11.sub(i))

        # norm of the initial non-linear residual
        norm_0 = vnorm(rhs)
        norm_k = norm_0

        with rhs_10.dat.vec as b_v:
            b_v.scale(tau)
        with rhs_11.dat.vec as b_v:
            b_v.scale(tau)

        k = 0

        print(f'Initial non-linear residual: {norm_0:.16e}')

        while (norm_k > relative_non_linear_tol * norm_0 and norm_k > absolute_non_linear_tol):
            # solving for the linearization
            self.incompressible_linear_solve(
                nullspace_p, space_p=space_p,
                P=P, solver_parameters=solver_parameters,
                auxiliary_sp=auxiliary_sp,
                v_d=rhs_00, f=rhs_01,
                div_v=rhs_10, div_zeta=rhs_11,
                print_error=False, create_output=False, plots=False)

            delta_v.assign(self._v)
            delta_zeta.assign(self._zeta)

            delta_p.assign(self._p)
            delta_mu.assign(self._mu)

            # udpating the solutions
            v_old += delta_v
            if inhomogeneous_bcs_v:
                for i in range(n_t):
                    apply_bcs(bcs_v_help[(i)], v_old.sub(i))

            p_old += delta_p

            zeta_old += delta_zeta
            for i in range(n_t):
                apply_bcs(bcs_zeta, zeta_old.sub(i))

            mu_old += delta_mu

            self.set_v(v_old)
            self.set_zeta(zeta_old)

            self.set_p(p_old)
            self.set_mu(mu_old)

            # evaluating the non-linear residual
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

            # norm of the non-linear residual
            norm_k = vnorm(rhs)

            with rhs_10.dat.vec as b_v:
                b_v.scale(tau)
            with rhs_11.dat.vec as b_v:
                b_v.scale(tau)

            k += 1

            print(f'Non-linear solver: '
                  f'iteration {k:d}, '
                  f'non-linear residual norm {norm_k:.16e}')

            if k + 1 > max_non_linear_iter:
                break

        if print_error_non_linear:
            if (norm_k < relative_non_linear_tol * norm_0 or norm_k < absolute_non_linear_tol):
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
                print(f'Number of non-linear iterations: {k:d}')
            else:
                print('The non-linear iteration did not converge')
                if norm_0 > 0.:
                    print(f'Relative non-linear residual: {norm_k / norm_0:.16e}')
                print(f'Absolute non-linear residual: {norm_k:.16e}')
            self.print_error(tau)

        if create_output:
            output({"v": self._v, "zeta": self._zeta, "p": self._p, "mu": self._mu})

        if plots:
            for i in range(n_t - 1):
                plot(self._v.sub(i), self._p.sub(i), self._zeta.sub(i), self._mu.sub(i), self._true_v.sub(i))
            if self._CN:
                plot(self._v.sub(n_t - 1), self._zeta.sub(n_t - 1), self._true_v.sub(n_t - 1))
            else:
                plot(self._v.sub(n_t - 1), self._p.sub(n_t - 1), self._zeta.sub(n_t - 1), self._mu.sub(n_t - 1), self._true_v.sub(n_t - 1))
