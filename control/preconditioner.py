"""A library for the construction of generalized saddle-point systems
arising from the finite element discretization of the KKT conditions of
a PDE constrained optimization problem. Uses the Python interface to PETSc
for the construction of the linear solver.
"""
from firedrake import (
    Cofunction, DirichletBC, Function, MixedFunctionSpace, assemble)
from firedrake.functionspaceimpl import WithGeometry as FunctionSpaceBase

from .cn import apply_T_1, apply_T_2

import petsc4py.PETSc as PETSc
import ufl

import abc
from collections.abc import Sequence


__all__ = \
    [
        "apply_bcs",

        "Nullspace",
        "NoneNullspace",
        "ConstantNullspace",
        "DirichletBCNullspace",
        "FullNullspace",

        "MultiBlockSystem"
    ]


def l2_riesz(u):
    if isinstance(u, Function):
        cls = Cofunction
    elif isinstance(u, Cofunction):
        cls = Function
    else:
        raise TypeError(f"Unexpected type: {type(u)}")
    return cls(u.function_space().dual(), val=u.dat)


def apply_bcs(bcs, u):
    if not isinstance(bcs, Sequence):
        bcs = (bcs,)
    for bc in bcs:
        if isinstance(bc.function_arg, Cofunction):
            bc = bc.reconstruct(g=l2_riesz(bc.function_arg))
        if isinstance(u, Cofunction) and isinstance(bc.function_arg, (ufl.classes.Zero, Function)):
            bc.apply(l2_riesz(u))
        else:
            bc.apply(u)


class Nullspace(abc.ABC):
    """
    Definition of the nullspace of an operator.

    """
    @abc.abstractmethod
    def apply_nullspace_transformation_lhs_right(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_nullspace_transformation_lhs_left(self, y):
        raise NotImplementedError

    @abc.abstractmethod
    def extended_correct_lhs(self, x, y):
        raise NotImplementedError

    @abc.abstractmethod
    def pc_extended_correct_soln(self, u, b):
        raise NotImplementedError

    def correct_soln(self, x):
        self.apply_nullspace_transformation_lhs_right(x)

    def pre_mult_corrected_lhs(self, x):
        x_c = Function(x.function_space())
        x_c.assign(x)
        self.apply_nullspace_transformation_lhs_right(x_c)
        return x_c

    def post_mult_correct_lhs(self, x, y):
        self.apply_nullspace_transformation_lhs_left(y)
        self.extended_correct_lhs(x, y)

    def correct_rhs(self, b):
        self.apply_nullspace_transformation_lhs_left(b)

    def pc_pre_mult_corrected(self, b):
        b_c = Function(b.function_space())
        b_c.assign(b)
        self.apply_nullspace_transformation_lhs_left(b_c)
        return b_c

    def pc_post_mult_correct(self, u, b):
        self.apply_nullspace_transformation_lhs_right(u)
        self.pc_extended_correct_soln(u, b)


class NoneNullspace(Nullspace):
    """
    No nullspace
    """
    def apply_nullspace_transformation_lhs_right(self, x):
        pass

    def apply_nullspace_transformation_lhs_left(self, y):
        pass

    def extended_correct_lhs(self, x, y):
        pass

    def pc_extended_correct_soln(self, u, b):
        pass


class ConstantNullspace(Nullspace):
    """
    Constant nullspace
    """
    def __init__(self, *, alpha=1.0):
        self._alpha = alpha

    def _correct(self, x, y, *, alpha=1.0):
        with x.dat.vec_ro as x_v:
            x_sum = x_v.sum()
            N = x_v.getSize()

        with y.dat.vec as y_v:
            y_v.shift(alpha * x_sum / float(N))

    def apply_nullspace_transformation_lhs_right(self, x):
        self._correct(x, x, alpha=-1.0)

    def apply_nullspace_transformation_lhs_left(self, y):
        self._correct(y, y, alpha=-1.0)

    def extended_correct_lhs(self, x, y):
        self._correct(x, y, alpha=self._alpha)

    def pc_extended_correct_soln(self, u, b):
        self._correct(b, u, alpha=1.0)


class DirichletBCNullspace(Nullspace):
    """
    Nulspace for Dirichlet boundary conditions
    """
    def __init__(self, bcs, *, alpha=1.0):
        if isinstance(bcs, Sequence):
            bcs = tuple(bcs)
        else:
            bcs = (bcs,)

        for bc in bcs:
            if not isinstance(bc.function_arg, ufl.classes.Zero):
                raise ValueError("Homogeneous boundary conditions required")

        self._bcs = bcs
        self._alpha = alpha

    def apply_nullspace_transformation_lhs_right(self, x):
        for bc in self._bcs:
            bc.apply(x)

    def apply_nullspace_transformation_lhs_left(self, y):
        apply_bcs(self._bcs, y)

    def _extended_correct_lhs(self, x, y, *, alpha=1.0):
        def bc_sub_domain_arg(bc):
            return (bc.sub_domain, )

        c = Function(x.function_space())
        for bc_0 in self._bcs:
            bc_1 = DirichletBC(bc_0.function_space(), x,
                               *bc_sub_domain_arg(bc_0))
            apply_bcs(bc_1, c)

        with c.dat.vec_ro as c_v, y.dat.vec as y_v:
            y_v.axpy(alpha, c_v)

    def extended_correct_lhs(self, x, y):
        self._extended_correct_lhs(x, y, alpha=self._alpha)

    def pc_extended_correct_soln(self, u, b):
        self._extended_correct_lhs(b, u, alpha=1.0)


class FullNullspace(Nullspace):
    """
    Full nullspace
    """
    def apply_nullspace_transformation_lhs_right(self, x):
        with x.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def apply_nullspace_transformation_lhs_left(self, y):
        with y.dat.vec_wo as y_v:
            y_v.zeroEntries()

    def extended_correct_lhs(self, x, y):
        y.assign(x)

    def pc_extended_correct_soln(self, u, b):
        u.assign(b)


class MultiBlockSystem:
    def __init__(self, space_0, space_1,
                 block_00, block_01, block_10, block_11, *,
                 n_blocks_00=1, n_blocks_11=1,
                 sub_n_blocks_00_0=None, sub_n_blocks_11_0=None,
                 nullspace_0=None, nullspace_1=None,
                 form_compiler_parameters=None, CN=False):
        """
        Class for the solution of generalized saddle-point system.

        Input:
            - space_0                     space to which the state solution
                                          belongs

            - space_1                     space to which the adjoint solution
                                          belongs

            - block_00                    dictionary representing the (1,1)-block
                                          of the generalized saddle-point system

            - block_01                    dictionary representing the (1,2)-block
                                          of the generalized saddle-point system

            - block_10                    dictionary representing the (2,1)-block
                                          of the generalized saddle-point system

            - block_11                    dictionary representing the (2,2)-block
                                          of the generalized saddle-point system

            - n_blocks_00                 number of sub-blocks composing the
                                          (1,1)-block

            - n_blocks_11                 number of sub-blocks composing the
                                          (2,2)-block

            - sub_n_blocks_00_0           for time-dependent problems when
                                          applying a trapezoidal rule in time,
                                          defines the number of sub-blocks of
                                          the (1,1)-block for applying the
                                          linear transformation employed in
                                          the derivation of the preconditioner

            - sub_n_blocks_11_0           for time-dependent problems when
                                          applying a trapezoidal rule in time,
                                          defines the number of sub-blocks of
                                          the (2,2)-block for applying the
                                          linear transformation employed in
                                          the derivation of the preconditioner

            - nullspace_0                 nullspace of the first component of
                                          the solution

            - nullspace_1                 nullspace of the second component of
                                          the solution

            - form_compiler_parameters    dictionary to be passed to the form
                                          compiler

            - CN                          for instationary problems, set True
                                          if employing a trapezoidal
                                          discretization in time
        """

        if nullspace_0 is None:
            nullspace_0 = ()
            for i in range(n_blocks_00):
                nullspace_0 = nullspace_0 + (NoneNullspace(), )
        if nullspace_1 is None:
            nullspace_1 = ()
            for i in range(n_blocks_11):
                nullspace_1 = nullspace_1 + (NoneNullspace(), )

        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        if space_0.mesh() != space_1.mesh():
            raise ValueError("Unexpected mesh")
        if not isinstance(space_0, FunctionSpaceBase):
            raise TypeError("Space must be a primal space")
        if not isinstance(space_1, FunctionSpaceBase):
            raise TypeError("Space must be a primal space")

        def check_blocks(block, n_row_blocks, n_col_blocks):
            if len(block) != n_row_blocks * n_col_blocks:
                raise ValueError("Unexpected dimension of blocks")

        def check_spaces(block, test_space, trial_space):
            test, trial = block.arguments()
            assert (test.number(), trial.number()) == (0, 1)
            if test.function_space() != test_space:
                raise ValueError("Unexpected test space")
            if trial.function_space() != trial_space:
                raise ValueError("Unexpected trial space")

        check_blocks(block_00, n_blocks_00, n_blocks_00)
        check_blocks(block_01, n_blocks_00, n_blocks_11)
        check_blocks(block_10, n_blocks_11, n_blocks_00)
        check_blocks(block_11, n_blocks_11, n_blocks_11)

        for (i, j), block_00_ij in block_00.items():
            if block_00_ij is not None:
                check_spaces(block_00_ij, space_0, space_0)
        for (i, j), block_01_ij in block_01.items():
            if block_01_ij is not None:
                check_spaces(block_01_ij, space_0, space_1)
        for (i, j), block_10_ij in block_10.items():
            if block_10_ij is not None:
                check_spaces(block_10_ij, space_1, space_0)
        for (i, j), block_11_ij in block_11.items():
            if block_11_ij is not None:
                check_spaces(block_11_ij, space_1, space_1)

        self._space_0 = space_0
        self._space_1 = space_1

        if n_blocks_00 == 1:
            flattened_space_0 = (space_0, )
        else:
            flattened_space_0 = tuple(space_0 for i in range(n_blocks_00))

        if n_blocks_11 == 1:
            flattened_space_1 = (space_1, )
        else:
            flattened_space_1 = tuple(space_1 for i in range(n_blocks_11))

        full_flattened_space = flattened_space_0 + flattened_space_1
        self._spaces = MixedFunctionSpace(full_flattened_space)

        self._n_blocks_00 = n_blocks_00
        self._n_blocks_11 = n_blocks_11
        if sub_n_blocks_00_0 is not None:
            sub_n_blocks_00_1 = n_blocks_00 - sub_n_blocks_00_0
        else:
            sub_n_blocks_00_1 = None
        if sub_n_blocks_11_0 is not None:
            sub_n_blocks_11_1 = n_blocks_11 - sub_n_blocks_11_0
        else:
            sub_n_blocks_11_1 = None
        self._sub_n_blocks_00_0 = sub_n_blocks_00_0
        self._sub_n_blocks_00_1 = sub_n_blocks_00_1
        self._sub_n_blocks_11_0 = sub_n_blocks_11_0
        self._sub_n_blocks_11_1 = sub_n_blocks_11_1
        self._blocks = (block_00, block_01, block_10, block_11)

        self._matrices_00 = {}
        for (i, j), block_ij in block_00.items():
            if block_ij is None:
                self._matrices_00[(i, j)] = None
            else:
                self._matrices_00[(i, j)] = assemble(block_ij, form_compiler_parameters=form_compiler_parameters)
        self._matrices_01 = {}
        for (i, j), block_ij in block_01.items():
            if block_ij is None:
                self._matrices_01[(i, j)] = None
            else:
                self._matrices_01[(i, j)] = assemble(block_ij, form_compiler_parameters=form_compiler_parameters)
        self._matrices_10 = {}
        for (i, j), block_ij in block_10.items():
            if block_ij is None:
                self._matrices_10[(i, j)] = None
            else:
                self._matrices_10[(i, j)] = assemble(block_ij, form_compiler_parameters=form_compiler_parameters)
        self._matrices_11 = {}
        for (i, j), block_ij in block_11.items():
            if block_ij is None:
                self._matrices_11[(i, j)] = None
            else:
                self._matrices_11[(i, j)] = assemble(block_ij, form_compiler_parameters=form_compiler_parameters)

        nullspaces = ()
        nullspaces = nullspaces + tuple(nullspace_0)
        nullspaces = nullspaces + tuple(nullspace_1)
        self._nullspaces = nullspaces

        self._CN = CN

    def solve(self, u_0, u_1, b_0, b_1, *,
              solver_parameters=None,
              pc_fn=None):
        """
        Definition of the linear solver.

        Input:
            - u_0                    first component of the solution

            - u_1                    second component of the solution

            - b_0                    first component of the right-hand side

            - b_1                    second component of the right-hand side

            - solver_parameters      dictionary, contains the parameters
                                     for the linear solver

            - pc_fn                  function defining the preconditioner
        """
        if solver_parameters is None:
            solver_parameters = {}
        if pc_fn is None:
            def pc_fn(u_0, u_1, b_0, b_1):
                u_0.assign(l2_riesz(b_0))
                u_1.assign(l2_riesz(b_1))

        class MultiBlockSystemMatrix:
            def __init__(self, n_blocks_00, n_blocks_11,
                         sub_n_blocks_00_0, sub_n_blocks_00_1,
                         sub_n_blocks_11_0, sub_n_blocks_11_1,
                         space_0, space_1, spaces,
                         matrices_00, matrices_01, matrices_10,
                         matrices_11, nullspaces, CN):
                """
                Definition of the generalized saddle-point system.

                Input:
                    - n_blocks_00            number of sub-blocks of the
                                             (1,1)-block

                    - n_blocks_11            number of sub-blocks of the
                                             (2,2)-block

                    - sub_n_blocks_00_0      for time-dependent problems when
                                             applying a trapezoidal rule in
                                             time, defines the number of
                                             sub-blocks of the (1,1)-block for
                                             applying the first linear
                                             transformation employed in the
                                             derivation of the preconditioner

                    - sub_n_blocks_00_1      for time-dependent problems when
                                             applying a trapezoidal rule in
                                             time, defines the number of
                                             sub-blocks of the (1,1)-block for
                                             applying the second linear
                                             transformation employed in the
                                             derivation of the preconditioner

                    - sub_n_blocks_11_0      for time dependent problems when
                                             applying a trapezoidal rule in
                                             time, defines the number of
                                             sub-blocks of the (2,2)-block for
                                             applying the first linear
                                             transformation employed in the
                                             derivation of the preconditioner

                    - sub_n_blocks_11_1      for time dependent problems when
                                             applying a trapezoidal rule in
                                             time, defines the number of
                                             sub-blocks of the (2,2)-block for
                                             applying the second linear
                                             transformation employed in the
                                             derivation of the preconditioner

                    - space_0                space to which the first component
                                             of the solution belongs

                    - space_1                space to which the second component
                                             of the solution belongs

                    - spaces                 space to which the full solution
                                             belongs

                    - matrices_00            dictionary representing the
                                             (1,1)-block of the generalized
                                             saddle-point system

                    - matrices_01            dictionary representing the
                                             (1,2)-block of the generalized
                                             saddle-point system

                    - matrices_10            dictionary representing the
                                             (2,1)-block of the generalized
                                             saddle-point system

                    - matrices_11            dictionary representing the
                                             (2,2)-block of the generalized
                                             saddle-point system

                    - nullspaces             nullspace of the full solution

                    - CN                     for instationary problems, set
                                             True if employing a trapezoidal
                                             discretization in time
                """
                self._space_0 = space_0
                self._space_1 = space_1
                self._spaces = spaces
                self._n_blocks_00 = n_blocks_00
                self._n_blocks_11 = n_blocks_11
                self._sub_n_blocks_00_0 = sub_n_blocks_00_0
                self._sub_n_blocks_00_1 = sub_n_blocks_00_1
                self._sub_n_blocks_11_0 = sub_n_blocks_11_0
                self._sub_n_blocks_11_1 = sub_n_blocks_11_1

                self._matrices_00 = matrices_00
                self._matrices_01 = matrices_01
                self._matrices_10 = matrices_10
                self._matrices_11 = matrices_11

                self._nullspaces = tuple(nullspaces)
                self._CN = CN

                self._x_fn = Function(spaces)
                self._y_fn = Function(spaces)

            def mult(self, A, x, y):
                """
                Definition of the matrix-vector multiplication

                Input:
                    - A        matrix representing the system

                    - x        vector of the solution

                    - y        right-hand side
                """
                with self._x_fn.dat.vec_wo as x_v:
                    x.copy(result=x_v)

                x_c_new = Function(self._spaces)

                x_c_0_help = Function(self._space_0)
                x_c_1_help = Function(self._space_1)
                for i in range(self._n_blocks_00):
                    x_c_0_help.assign(self._x_fn.sub(i))
                    nullspace_help = self._nullspaces[i]
                    x_c_0_i = nullspace_help.pre_mult_corrected_lhs(x_c_0_help)
                    x_c_new.sub(i).assign(x_c_0_i)
                for i in range(self._n_blocks_11):
                    x_c_1_help.assign(self._x_fn.sub(self._n_blocks_00 + i))
                    nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                    x_c_1_i = nullspace_help.pre_mult_corrected_lhs(x_c_1_help)
                    x_c_new.sub(self._n_blocks_00 + i).assign(x_c_1_i)

                for i in range(self._n_blocks_00 + self._n_blocks_11):
                    with self._y_fn.sub(i).dat.vec_wo as y_0_v:
                        y_0_v.zeroEntries()

                x_help_0 = Function(self._space_0)
                x_help_1 = Function(self._space_1)
                for (i, j), block_ij in self._matrices_00.items():
                    if block_ij is not None:
                        x_help_0.assign(x_c_new.sub(j))
                        with x_help_0.dat.vec_ro as x_0_v, \
                                self._y_fn.sub(i).dat.vec as y_0_v:
                            block_ij.petscmat.multAdd(x_0_v, y_0_v, y_0_v)

                for (i, j), block_ij in self._matrices_01.items():
                    if block_ij is not None:
                        x_help_1.assign(x_c_new.sub(self._n_blocks_00 + j))
                        with x_help_1.dat.vec_ro as x_0_v, \
                                self._y_fn.sub(i).dat.vec as y_0_v:
                            block_ij.petscmat.multAdd(x_0_v, y_0_v, y_0_v)

                for (i, j), block_ij in self._matrices_10.items():
                    if block_ij is not None:
                        x_help_0.assign(x_c_new.sub(j))
                        with x_help_0.dat.vec_ro as x_0_v, \
                                self._y_fn.sub(self._n_blocks_00 + i).dat.vec as y_0_v:
                            block_ij.petscmat.multAdd(x_0_v, y_0_v, y_0_v)

                for (i, j), block_ij in self._matrices_11.items():
                    if block_ij is not None:
                        x_help_1.assign(x_c_new.sub(self._n_blocks_00 + j))
                        with x_help_1.dat.vec_ro as x_0_v, \
                                self._y_fn.sub(self._n_blocks_00 + i).dat.vec as y_0_v:
                            block_ij.petscmat.multAdd(x_0_v, y_0_v, y_0_v)

                if self._CN:
                    if self._sub_n_blocks_00_0 is None and self._sub_n_blocks_11_0 is None:
                        if self._n_blocks_00 == 1:
                            space_0_help = self._space_0
                        else:
                            flattened_space_0 = tuple(self._space_0 for i in range(self._n_blocks_00))
                            space_0_help = MixedFunctionSpace(flattened_space_0)

                        if self._n_blocks_11 == 1:
                            space_1_help = self._space_1
                        else:
                            flattened_space_1 = tuple(self._space_1 for i in range(self._n_blocks_11))
                            space_1_help = MixedFunctionSpace(flattened_space_1)

                        y_help_0 = Function(space_0_help)
                        y_help_1 = Function(space_1_help)

                        for i in range(self._n_blocks_00):
                            y_help_0.sub(i).assign(self._y_fn.sub(i))
                        for i in range(self._n_blocks_11):
                            index = self._n_blocks_00 + i
                            y_help_1.sub(i).assign(self._y_fn.sub(index))

                        y_help_0 = apply_T_1(y_help_0, self._space_0, self._n_blocks_00)
                        y_help_1 = apply_T_2(y_help_1, self._space_1, self._n_blocks_11)

                        for i in range(self._n_blocks_00):
                            self._y_fn.sub(i).assign(y_help_0.sub(i))
                        for i in range(self._n_blocks_11):
                            index = self._n_blocks_00 + i
                            self._y_fn.sub(index).assign(y_help_1.sub(i))
                    else:
                        flattened_space_0_0 = tuple(self._space_0 for i in range(self._sub_n_blocks_00_0))
                        space_0_0_help = MixedFunctionSpace(
                            flattened_space_0_0)

                        flattened_space_0_1 = tuple(self._space_0 for i in range(self._sub_n_blocks_00_1))
                        space_0_1_help = MixedFunctionSpace(
                            flattened_space_0_1)

                        flattened_space_1_0 = tuple(self._space_1 for i in range(self._sub_n_blocks_11_0))
                        space_1_0_help = MixedFunctionSpace(
                            flattened_space_1_0)

                        flattened_space_1_1 = tuple(self._space_1 for i in range(self._sub_n_blocks_11_1))
                        space_1_1_help = MixedFunctionSpace(
                            flattened_space_1_1)

                        y_help_0_0 = Function(space_0_0_help)
                        y_help_0_1 = Function(space_0_1_help)
                        y_help_1_0 = Function(space_1_0_help)
                        y_help_1_1 = Function(space_1_1_help)

                        for i in range(self._sub_n_blocks_00_0):
                            y_help_0_0.sub(i).assign(self._y_fn.sub(i))
                        for i in range(self._sub_n_blocks_00_1):
                            index = self._sub_n_blocks_00_0 + i
                            y_help_0_1.sub(i).assign(self._y_fn.sub(index))
                        for i in range(self._sub_n_blocks_11_0):
                            index = self._n_blocks_00 + i
                            y_help_1_0.sub(i).assign(self._y_fn.sub(index))
                        for i in range(self._sub_n_blocks_11_1):
                            index = self._n_blocks_00 + self._sub_n_blocks_11_0 + i
                            y_help_1_1.sub(i).assign(self._y_fn.sub(index))

                        y_help_0_0 = apply_T_1(y_help_0_0, self._space_0, self._sub_n_blocks_00_0)
                        y_help_0_1 = apply_T_2(y_help_0_1, self._space_0, self._sub_n_blocks_00_1)
                        y_help_1_0 = apply_T_2(y_help_1_0, self._space_1, self._sub_n_blocks_11_0)
                        y_help_1_1 = apply_T_1(y_help_1_1, self._space_1, self._sub_n_blocks_11_1)

                        for i in range(self._sub_n_blocks_00_0):
                            self._y_fn.sub(i).assign(y_help_0_0.sub(i))
                        for i in range(self._sub_n_blocks_00_1):
                            index = self._sub_n_blocks_00_0 + i
                            self._y_fn.sub(index).assign(y_help_0_1.sub(i))
                        for i in range(self._sub_n_blocks_11_0):
                            index = self._n_blocks_00 + i
                            self._y_fn.sub(index).assign(y_help_1_0.sub(i))
                        for i in range(self._sub_n_blocks_11_1):
                            index = self._n_blocks_00 + self._sub_n_blocks_11_0 + i
                            self._y_fn.sub(index).assign(y_help_1_1.sub(i))

                x_c_0_help = Function(self._space_0)
                x_c_1_help = Function(self._space_1)

                for i in range(self._n_blocks_00):
                    x_c_0_help.assign(self._x_fn.sub(i))
                    nullspace_help = self._nullspaces[i]
                    nullspace_help.post_mult_correct_lhs(x_c_0_help, self._y_fn.sub(i))
                for i in range(self._n_blocks_11):
                    x_c_1_help.assign(self._x_fn.sub(self._n_blocks_00 + i))
                    nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                    nullspace_help.post_mult_correct_lhs(x_c_1_help, self._y_fn.sub(self._n_blocks_00 + i))

                with self._y_fn.dat.vec_ro as y_v:
                    y_v.copy(result=y)

        class Preconditioner:
            def __init__(self, n_blocks_00, n_blocks_11,
                         space_0, space_1, spaces,
                         pc_fn, nullspaces):
                """
                Definition of the preconditioner.

                Input:
                    - n_blocks_00        number of sub-blocks of the
                                         (1,1)-block

                    - n_blocks_11        number of sub-blocks of the
                                         (2,2)-block

                    - space_0            space to which the first component
                                         of the solution belongs

                    - space_1            space to which the second component
                                         of the solution belongs

                    - spaces             space to which the full solution
                                         belongs

                    - pc_fn              function defining the preconditioner

                    - nullspaces         nullspace of the full solution
                """
                self._pc_fn = pc_fn
                self._n_blocks_00 = n_blocks_00
                self._n_blocks_11 = n_blocks_11
                self._nullspaces = tuple(nullspaces)

                self._x_fn = Cofunction(spaces.dual())
                self._y_fn = Function(spaces)

                self._space_0 = space_0
                self._space_1 = space_1

            def apply(self, pc, x, y):
                """
                Definition of the application of the preconditioner.

                Input:
                    - pc        preconditioner

                    - x         vector of the solution

                    - y         right-hand side
                """
                with self._x_fn.dat.vec_wo as x_v:
                    x.copy(result=x_v)

                space_0 = self._space_0
                space_1 = self._space_1

                if self._n_blocks_00 == 1:
                    space_help_0 = self._space_0
                else:
                    flattened_space_0 = tuple(space_0 for i in range(self._n_blocks_00))
                    space_help_0 = MixedFunctionSpace(flattened_space_0)

                if self._n_blocks_11 == 1:
                    space_help_1 = self._space_1
                else:
                    flattened_space_1 = tuple(space_1 for i in range(self._n_blocks_11))
                    space_help_1 = MixedFunctionSpace(flattened_space_1)

                b_0 = Cofunction(space_help_0.dual(), name="b_0")
                b_1 = Cofunction(space_help_1.dual(), name="b_1")
                if self._n_blocks_00 == 1:
                    b_0.assign(self._x_fn.sub(0))
                else:
                    for i in range(self._n_blocks_00):
                        b_0.sub(i).assign(self._x_fn.sub(i))

                if self._n_blocks_11 == 1:
                    b_1.assign(self._x_fn.sub(self._n_blocks_00))
                else:
                    for i in range(self._n_blocks_11):
                        b_1.sub(i).assign(self._x_fn.sub(self._n_blocks_00 + i))

                b_0_c = Cofunction(space_help_0.dual(), name="b_0_c")
                b_1_c = Cofunction(space_help_1.dual(), name="b_1_c")
                if self._n_blocks_00 == 1:
                    nullspace_help = self._nullspaces[0]
                    b_0_c.assign(nullspace_help.pc_pre_mult_corrected(b_0))
                else:
                    b_0_c_help = Cofunction(space_0.dual())
                    for i in range(self._n_blocks_00):
                        b_0_c_help.assign(b_0.sub(i))
                        nullspace_help = self._nullspaces[i]
                        b_0_c.sub(i).assign(nullspace_help.pc_pre_mult_corrected(b_0_c_help))

                if self._n_blocks_11 == 1:
                    nullspace_help = self._nullspaces[self._n_blocks_00]
                    b_1_c.assign(nullspace_help.pc_pre_mult_corrected(b_1))
                else:
                    b_1_c_help = Cofunction(space_1.dual())
                    for i in range(self._n_blocks_11):
                        b_1_c_help.assign(b_1.sub(i))
                        nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                        b_1_c.sub(i).assign(nullspace_help.pc_pre_mult_corrected(b_1_c_help))

                u_0 = Function(space_help_0, name="u_0")
                u_1 = Function(space_help_1, name="u_1")

                pc_fn(u_0, u_1, b_0_c, b_1_c)

                if self._n_blocks_00 == 1:
                    self._y_fn.sub(0).assign(u_0)
                    nullspace_help = self._nullspaces[0]
                    nullspace_help.pc_post_mult_correct(self._y_fn.sub(0), b_0)
                else:
                    y_c_0_help = Cofunction(space_0.dual())
                    for i in range(self._n_blocks_00):
                        y_c_0_help.assign(b_0.sub(i))
                        self._y_fn.sub(i).assign(u_0.sub(i))
                        nullspace_help = self._nullspaces[i]
                        nullspace_help.pc_post_mult_correct(self._y_fn.sub(i), y_c_0_help)

                if self._n_blocks_11 == 1:
                    self._y_fn.sub(self._n_blocks_00).assign(u_1)
                    nullspace_help = self._nullspaces[self._n_blocks_00]
                    nullspace_help.pc_post_mult_correct(self._y_fn.sub(self._n_blocks_00), b_1)
                else:
                    y_c_1_help = Cofunction(space_1.dual())
                    for i in range(self._n_blocks_11):
                        y_c_1_help.assign(b_1.sub(i))
                        self._y_fn.sub(self._n_blocks_00 + i).assign(u_1.sub(i))
                        nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                        nullspace_help.pc_post_mult_correct(self._y_fn.sub(self._n_blocks_00 + i), y_c_1_help)

                with self._y_fn.dat.vec_ro as y_v:
                    y_v.copy(result=y)

        u = Function(self._spaces, name="u")

        if self._n_blocks_00 == 1:
            u.sub(0).assign(u_0)
            nullspace_help = self._nullspaces[0]
            nullspace_help.correct_soln(u.sub(0))
        else:
            for i in range(self._n_blocks_00):
                u.sub(i).assign(u_0.sub(i))
                nullspace_help = self._nullspaces[i]
                nullspace_help.correct_soln(u.sub(i))

        if self._n_blocks_11 == 1:
            u.sub(self._n_blocks_00).assign(u_1)
            nullspace_help = self._nullspaces[self._n_blocks_00]
            nullspace_help.correct_soln(u.sub(self._n_blocks_00))
        else:
            for i in range(self._n_blocks_11):
                u.sub(self._n_blocks_00 + i).assign(u_1.sub(i))
                nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                nullspace_help.correct_soln(u.sub(self._n_blocks_00 + i))

        b = Cofunction(self._spaces.dual())

        if self._n_blocks_00 == 1:
            if b_0 is not None:
                b.sub(0).assign(b_0)
                nullspace_help = self._nullspaces[0]
                nullspace_help.correct_rhs(b.sub(0))
        else:
            for i in range(self._n_blocks_00):
                if b_0.sub(i) is not None:
                    b.sub(i).assign(b_0.sub(i))
                    nullspace_help = self._nullspaces[i]
                    nullspace_help.correct_rhs(b.sub(i))

        if self._n_blocks_11 == 1:
            if b_1 is not None:
                b.sub(self._n_blocks_00).assign(b_1)
                nullspace_help = self._nullspaces[self._n_blocks_00]
                nullspace_help.correct_rhs(b.sub(self._n_blocks_00))
        else:
            for i in range(self._n_blocks_11):
                if b_1.sub(i) is not None:
                    b.sub(self._n_blocks_00 + i).assign(b_1.sub(i))
                    nullspace_help = self._nullspaces[self._n_blocks_00 + i]
                    nullspace_help.correct_rhs(b.sub(self._n_blocks_00 + i))

        comm = self._spaces.mesh().comm
        with b.dat.vec_ro as b_v:
            n = b_v.getLocalSize()
            N = b_v.getSize()
        A = MultiBlockSystemMatrix(self._n_blocks_00, self._n_blocks_11,
                                   self._sub_n_blocks_00_0,
                                   self._sub_n_blocks_00_1,
                                   self._sub_n_blocks_11_0,
                                   self._sub_n_blocks_11_1,
                                   self._space_0, self._space_1, self._spaces,
                                   self._matrices_00, self._matrices_01,
                                   self._matrices_10, self._matrices_11,
                                   self._nullspaces, self._CN)

        mat = PETSc.Mat().createPython(
            ((n, N), (n, N)), A, comm=comm)
        mat.setUp()

        pc = PETSc.PC().createPython(
            Preconditioner(self._n_blocks_00, self._n_blocks_11,
                           self._space_0, self._space_1, self._spaces,
                           pc_fn, self._nullspaces),
            comm=comm)
        pc.setOperators(mat)
        pc.setUp()

        ksp_solver = PETSc.KSP().create(comm=comm)
        ksp_solver.setType(solver_parameters.get("linear_solver", "fgmres"))
        ksp_solver.setPC(pc)
        if "pc_side" in solver_parameters:
            ksp_solver.setPCSide(solver_parameters["pc_side"])
        ksp_solver.setOperators(mat)
        ksp_solver.setTolerances(
            rtol=solver_parameters["relative_tolerance"],
            atol=solver_parameters["absolute_tolerance"],
            divtol=solver_parameters.get("divergence limit", None),
            max_it=solver_parameters.get("maximum_iterations", 1000))
        ksp_solver.setInitialGuessNonzero(True)
        ksp_solver.setNormType(
            solver_parameters.get(
                "norm_type", PETSc.KSP.NormType.DEFAULT))
        if "gmres_restart" in solver_parameters:
            ksp_solver.setGMRESRestart(solver_parameters["gmres_restart"])
        if solver_parameters.get("monitor_convergence", True):
            def monitor(ksp_solver, it, r_norm):
                print(f"KSP: "
                      f"iteration {it:d}, "
                      f"residual norm {r_norm:.16e}")
            ksp_solver.setMonitor(monitor)
        ksp_solver.setUp()
        precond = solver_parameters.get("preconditioner", False)

        with u.dat.vec as u_v, b.dat.vec as b_v:
            ksp_solver.solve(b_v, u_v)

        for i in range(self._n_blocks_00):
            nullspace_help = self._nullspaces[i]
            nullspace_help.correct_soln(u.sub(i))
        for i in range(self._n_blocks_11):
            nullspace_help = self._nullspaces[self._n_blocks_00 + i]
            nullspace_help.correct_soln(u.sub(self._n_blocks_00 + i))

        reason = ksp_solver.getConvergedReason()
        if precond:
            if reason <= 0 and reason != PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT:
                raise RuntimeError("Solver failed to converge")
        else:
            if reason <= 0:
                raise RuntimeError("Solver failed to converge")

        if self._n_blocks_00 == 1:
            u_0.assign(u.sub(0))
        else:
            for i in range(self._n_blocks_00):
                u_0.sub(i).assign(u.sub(i))

        if self._n_blocks_11 == 1:
            u_1.assign(u.sub(self._n_blocks_00))
        else:
            for i in range(self._n_blocks_11):
                u_1.sub(i).assign(u.sub(self._n_blocks_00 + i))

        return ksp_solver
