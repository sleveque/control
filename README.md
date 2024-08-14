control is a library for solving certain PDE-constrained optimization problems. The software employs the Firedrake
system to derive the finite element discretization of the problems considered, using the Python interface to PETSc
for the derivation of the KKT conditions and the definition of the linear solvers.

For simplicity, we consider the following heat control problem:

$\min_{v, u} \frac{1}{2} \int_0^{t_f} \| v - v_d \|^2_{L^2(\Omega)} \mathrm{d} t + \frac{\beta}{2} \int_0^{t_f} \| u \|^2_{L^2(\Omega)} \mathrm{d} t$

subject to

$\left\{ \begin{array}{ll} \frac{\partial v}{\partial t} -\nabla^2 v = u + f, & \mathrm{in} \; \Omega \times (0, t_f)\\ v(\mathbf{x},0)=0, & \mathrm{in} \; \Omega\\ v(\mathbf{x},t)=0, & \mathrm{on} \; \partial \Omega \times(0, t_f), \end{array} \right.$

where, for example, $\Omega = (0, 1)^2$, $\beta = 10^{-4}$, and $t_f = 2$.

The problem is defined in a compact way by providing the weak form representing the forward differential operator
in space, the boundary conditions on the state variable, the desired state, and the force function acting on the
system. Note that, for stationary control problems, the callables defining the desired state and the force function
accept as an input the test function of the finite element space considered, while the callable related to the
forward differential operator accepts as inputs the trial function, the test function, and the current approximation
of the state $v$. For instationary problems one has to include also the time $t$. Finally the problem is defined by
instantiating an \texttt{Instationary} object. For the example considered, the code looks as follows:

	from firedrake import *
	from preconditioner import *
	from control import *

	mesh = UnitSquareMesh(10, 10, 2.0, 2.0)
	space_0 = FunctionSpace(mesh, "Lagrange", 1)

	def forw_diff_operator(trial, test, v, t):
	    return inner(grad(trial), grad(test)) * dx

	def desired_state(test, t):
	    space = test.function_space()
	    mesh = space.mesh()
	    X = SpatialCoordinate(mesh)
	    x = X[0] - 1.0
	    y = X[1] - 1.0

	    v_d = Function(space, name="v_d")
	    v_d.interpolate(t * cos(0.5 * pi * x) cos(0.5 * pi * y))

	    return inner(v_d, test) * dx, v_d

	def force_f(test, t):
	    space = test.function_space()
	    mesh = space.mesh()
	    X = SpatialCoordinate(mesh)
	    x = X[0] - 1.0
	    y = X[1] - 1.0

	    f = Function(space, name="f")
	    f.interpolate(cos(0.5 * pi * x) cos(0.5 * pi * y))

	    return inner(f, test) * dx

	def bc_t(space_0, t):
		return DirichletBC(space_0, 0.0, "on_boundary")

	control_instationary = Control.Instationary(
	    space_0, forw_diff_operator, desired_state=desired_state,
	    force_f=force_f, bcs_v=bc_t, beta=1.0e-4, n_t=10,
	    time_interval=(0.0, 2.0))

For instationary problems, the user can also provide a callable for the definition of a different initial condition,
passing the argument \texttt{initial\_condition}. The discretization in time can be set to backward Euler by passing
the argument \texttt{CN = False} to the call; the default discretization in time is trapezi.

Control problems which include incompressibility constraints are defined by passing to the extra argument
\texttt{space\_p}, the space to which the pressure belongs. The software assumes that inf--sup stable finite
element pairs are used.

The linear solvers consist of a Krylov solver preconditioned by suitable approximations of the matrices considered.
The solvers allow one to employ a user-defined preconditioner, adding to the call the argument \texttt{P} and passing
the solver parameters through the extra argument \texttt{solver\_parameters}. The non-linear solver is based on a
Picard iteration, but it can be set to a Gauss--Newton method by passing the argument \texttt{Gauss\_Newton = True}
to the definition of the object. In the following section, we give details of the in-built preconditioners employed
in our software.
