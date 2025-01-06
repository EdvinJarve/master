import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import os

class FEMBase:
    def __init__(self, mesh, T, dt, initial_v=0.0):
        self.mesh = mesh
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)
        
        # Create a function space
        self.V = fem.functionspace(mesh, ("Lagrange", 1))
        
        # Initialize solution functions
        self.v_n = fem.Function(self.V)
        self.v_n.name = "v_n"
        
        self.v_h = fem.Function(self.V)
        self.v_h.name = "v_h"
        
        # Initialize v_n and v_h with initial_v, which can be a scalar or a function
        self.initialize_v(initial_v)
        
        # Additional initializations can go here
    
    def initialize_v(self, initial_v):
        """
        Initialize the membrane potential v_n and v_h with initial_v, which can be a scalar or a function.
        """
        if callable(initial_v):
            # initial_v is a function of spatial coordinates x
            self.v_n.interpolate(initial_v)
            self.v_h.interpolate(initial_v)
        else:
            # initial_v is a scalar value
            self.v_n.interpolate(lambda x: np.full_like(x[0], initial_v))
            self.v_h.interpolate(lambda x: np.full_like(x[0], initial_v))

    def initialize_ode_variables(self):
        """
        Initialize ODE variables with initial conditions, which can be scalars or functions.
        """
        num_nodes = len(self.v_n.x.array)
        self.num_ode_vars = len(self.ode_initial_conditions)

        self.ode_vars = np.zeros((num_nodes, self.num_ode_vars))
        for i, init_cond in enumerate(self.ode_initial_conditions):
            if callable(init_cond):
                # Initialize with a spatially varying function
                x = self.V.tabulate_dof_coordinates().T  # Shape (2, num_nodes)
                self.ode_vars[:, i] = init_cond(x)
            else:
                # Initialize with a scalar value
                self.ode_vars[:, i] = init_cond
            
class MonodomainSolverFEM(FEMBase):
    def __init__(self, mesh, T, dt, M_i, source_term_func=None, ode_system=None, ode_initial_conditions=None, initial_v=0.0):
        super().__init__(mesh, T, dt, initial_v)
        
        self.M_i = M_i  # Conductivity tensor
        self.time = fem.Constant(self.mesh, PETSc.ScalarType(0.0))  # Time as UFL Constant
        self.source_term_func = source_term_func  # Explicit source term
        self.ode_system = ode_system  # ODE system for current term
        self.ode_initial_conditions = ode_initial_conditions
        
        # Set up the variational problem
        self.create_variational_problem()
        self.create_solver()
        
        # If ODE system is provided, initialize ODE variables
        if self.ode_system is not None:
            self.initialize_ode_variables()
        
    
    def create_variational_problem(self):
        # Set up trial and test functions
        self.u_v, self.v_v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.mesh)
        
        # Bilinear form
        self.a_v = (self.u_v * self.v_v + self.dt * ufl.dot(self.M_i * ufl.grad(self.u_v), ufl.grad(self.v_v))) * dx
        
        # Right-hand side will be defined in the run method
        self.bilinear_form_v = fem.form(self.a_v)
        
        # Assemble matrix
        self.A_v = fem.petsc.assemble_matrix(self.bilinear_form_v)
        self.A_v.assemble()
        
        # Create vector for RHS
        self.b_v = fem.petsc.create_vector(fem.form(self.a_v))
        
    def create_solver(self):
        # Create linear solver
        self.solver_v = PETSc.KSP().create(self.mesh.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)
        
    def initialize_ode_variables(self):
        # Initialize ODE variables
        num_nodes = len(self.v_n.x.array)
        self.num_ode_vars = len(self.ode_initial_conditions)
        self.ode_vars = np.tile(self.ode_initial_conditions, (num_nodes, 1))
        
    def run(self, analytical_solution_v=None, time_points=None):
        errors_v = []
        start_time = time.time()
        
        # Get coordinates for error computation if needed
        if analytical_solution_v is not None or time_points is not None:
            dof_coords = self.V.tabulate_dof_coordinates()
            x_coords = dof_coords[:, 0]
            y_coords = dof_coords[:, 1]
        
        # Initialize solutions dictionary if time_points are provided
        if time_points is not None:
            solutions = {t: None for t in time_points}
        
        # Initial ODE state
        if self.ode_system is not None:
            v_values = self.v_n.x.array.copy()
            s_values = self.ode_vars[:, 1:] if self.num_ode_vars > 1 else np.empty((len(v_values), 0))
        
        for i in range(self.num_steps + 1):
            t = i * self.dt
            self.time.value = t  # Update the UFL Constant time value
            
            if self.ode_system is not None:
                # Solve ODE for half-step
                v_values, s_values = self.solve_ode(t, t + self.dt / 2, v_values, s_values)
                self.v_n.x.array[:] = v_values
                
                # Set up RHS of PDE using v_n
                self.setup_rhs_pde()
                
                # Solve PDE
                self.solve_pde()
                
                # Solve ODE for second half-step
                v_values = self.v_h.x.array.copy()
                v_values, s_values = self.solve_ode(t + self.dt / 2, t + self.dt, v_values, s_values)
                self.v_n.x.array[:] = v_values
            else:
                # Set up RHS of PDE with explicit source term
                self.setup_rhs_pde()
                
                # Solve PDE
                self.solve_pde()
                
                # Update v_n for next time step
                self.v_n.x.array[:] = self.v_h.x.array
                
            # Store the solution if the current time is one of the specified time points
            if time_points is not None and np.isclose(t, time_points, atol=1e-5).any():
                solutions[t] = self.v_h.x.array.copy()
            
            # Error computation if analytical_solution_v is provided
            if analytical_solution_v is not None:
                analytical_values_v = analytical_solution_v(x_coords, y_coords, t)
                numerical_values_v = self.v_h.x.array
                error_v = np.sqrt(np.sum((numerical_values_v - analytical_values_v) ** 2) / len(analytical_values_v))
                errors_v.append(error_v)
        
        computation_time = time.time() - start_time
    
        # Return errors, computation time, and solutions if time_points are provided
        if time_points is not None:
            return errors_v, computation_time, solutions
        else:
            return errors_v, computation_time

        
    def setup_rhs_pde(self):
        dx = ufl.Measure("dx", domain=self.mesh)
        if self.ode_system is not None:
            # No explicit source term; RHS is just v_n
            self.L_v = self.v_n * self.v_v * dx
        else:
            # Use explicit source term
            x_coord, y_coord = ufl.SpatialCoordinate(self.mesh)
            I_app = self.source_term_func(x_coord, y_coord, self.time)
            self.L_v = (self.v_n + self.dt * I_app) * self.v_v * dx
                
        self.linear_form_v = fem.form(self.L_v)
        with self.b_v.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_v, self.linear_form_v)

        
    def solve_pde(self):
        self.solver_v.solve(self.b_v, self.v_h.vector)
        self.v_h.x.scatter_forward()
        
    def solve_ode(self, t_start, t_end, v_values, s_values):
        num_nodes = len(v_values)
        def ode_system_vectorized(t, y_flat):
            y = y_flat.reshape(num_nodes, self.num_ode_vars)
            v = y[:, 0]
            s = y[:, 1:] if self.num_ode_vars > 1 else np.empty((num_nodes, 0))
            dv_dt, ds_dt = self.ode_system(t, v, s)
            dy_dt = np.hstack((dv_dt[:, None], ds_dt)) if ds_dt.size > 0 else dv_dt[:, None]
            return dy_dt.flatten()
        
        y0 = np.hstack((v_values[:, None], s_values)).flatten()
        sol = solve_ivp(ode_system_vectorized, [t_start, t_end], y0, method='RK45')
        y_end = sol.y[:, -1]
        y_end = y_end.reshape(num_nodes, self.num_ode_vars)
        v_new = y_end[:, 0]
        s_new = y_end[:, 1:] if self.num_ode_vars > 1 else np.empty((num_nodes, 0))
        return v_new, s_new

class MonodomainSolverFEMHardcode(FEMBase):
    def __init__(
        self,
        mesh,
        T,
        dt,
        M_i,
        source_term_func=None,
        ode_system=None,
        initial_v=0.0,
        initial_s=0.0,  # Added initial condition for s (e.g., w)
        theta=0.5        # Splitting parameter (0.5 for Strang splitting)
    ):
        """
        Initialize the MonodomainSolverFEMHardcode with hardcoded ODE initial conditions for w.

        Parameters:
            mesh: The mesh object.
            T (float): Total simulation time.
            dt (float): Time step size.
            M_i (float or array): Conductivity tensor.
            source_term_func (callable, optional): Function for explicit source term.
            ode_system (callable, optional): Function defining the ODE system.
            initial_v (float or callable, optional): Initial membrane potential.
        """
        # Hardcode the ODE initial conditions for the FitzHugh-Nagumo model (w = 0.0)
        ode_initial_conditions = [initial_s]  # Only w is part of the ODE
        
        super().__init__(
            mesh=mesh,
            T=T,
            dt=dt,
            initial_v=initial_v
        )
        
        self.M_i = M_i  # Conductivity tensor
        self.time = fem.Constant(self.mesh, PETSc.ScalarType(0.0))  # Time as UFL Constant
        self.source_term_func = source_term_func  # Explicit source term
        self.ode_system = ode_system  # ODE system for current term
        self.ode_initial_conditions = ode_initial_conditions  # Hardcoded
        self.theta = theta  # Splitting parameter
        
        # Set up the variational problem
        self.create_variational_problem()
        self.create_solver()
        
        # If ODE system is provided, initialize ODE variables
        if self.ode_system is not None:
            self.initialize_ode_variables()
    
    def initialize_ode_variables(self):
        """
        Initialize ODE variables with initial conditions, which can be scalars or functions.
        """
        num_nodes = self.v_n.x.array.size
        self.num_ode_vars = len(self.ode_initial_conditions)  # Number of state variables
        self.ode_vars = np.zeros((num_nodes, self.num_ode_vars))
        for i, init_cond in enumerate(self.ode_initial_conditions):
            if callable(init_cond):
                # Initialize with a spatially varying function
                x = self.V.tabulate_dof_coordinates()
                # Handle 2D coordinates: x is (num_nodes, 2)
                self.ode_vars[:, i] = init_cond(x.T)  # Pass (2, num_nodes)
            else:
                # Initialize with a scalar value
                self.ode_vars[:, i] = init_cond
    
    def create_variational_problem(self):
        # Set up trial and test functions
        self.u_v, self.v_v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.mesh)
        
        # Bilinear form: (u_v * v_v + dt * (M_i * grad(u_v) . grad(v_v))) * dx
        self.a_v = (self.u_v * self.v_v + self.dt * ufl.dot(self.M_i * ufl.grad(self.u_v), ufl.grad(self.v_v))) * dx
        
        # Right-hand side will be defined in the run method
        self.bilinear_form_v = fem.form(self.a_v)
        
        # Assemble matrix
        self.A_v = fem.petsc.assemble_matrix(self.bilinear_form_v)
        self.A_v.assemble()
        
        # Create vector for RHS
        self.b_v = fem.petsc.create_vector(self.bilinear_form_v)
    
    def create_solver(self):
        # Create linear solver
        self.solver_v = PETSc.KSP().create(self.mesh.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)
    
    def run(self, analytical_solution_v=None, time_points=None):
        errors_v = []
        start_time = time.time()
        
        # Get coordinates for error computation if needed
        if analytical_solution_v is not None or time_points is not None:
            dof_coords = self.V.tabulate_dof_coordinates()
            x_coords = dof_coords[:, 0]
            y_coords = dof_coords[:, 1]
        
        # Initialize solutions dictionary if time_points are provided
        if time_points is not None:
            solutions = {t: None for t in time_points}
        
        # Initial ODE state
        if self.ode_system is not None:
            v_values = self.v_n.x.array.copy()
            s_values = self.ode_vars[:, 0].copy()
        
        for i in range(self.num_steps + 1):
            t = i * self.dt
            self.time.value = t  # Update the UFL Constant time value

            if self.ode_system is not None:
                # Step 1: ODE Step (First Half-Step)
                v_values, s_values = self.solve_ode(t, t + self.theta * self.dt, v_values, s_values)
                self.v_n.x.array[:] = v_values

                # Step 2: PDE Step (Full Step)
                self.setup_rhs_pde()
                self.solve_pde()
                v_new = self.v_h.x.array.copy()

                # Step 3: ODE Step (Second Half-Step)
                v_values, s_values = self.solve_ode(t + self.theta * self.dt, t + self.dt, v_new, s_values)
                self.v_n.x.array[:] = v_values
                self.ode_vars[:, 0] = s_values
            else:
                # No ODE system; solve PDE with explicit source term
                self.setup_rhs_pde()
                self.solve_pde()

                # Update v_n for next time step
                v_values = self.v_h.x.array.copy()
                self.v_n.x.array[:] = v_values

            # Store the solution if the current time is one of the specified time points
            if time_points is not None and np.isclose(t, time_points, atol=1e-8).any():
                solutions[t] = self.v_n.x.array.copy()

            # Error computation if analytical_solution_v is provided
            if analytical_solution_v is not None:
                analytical_values_v = analytical_solution_v(x_coords, y_coords, t)
                numerical_values_v = self.v_n.x.array
                error_v = np.sqrt(np.mean((numerical_values_v - analytical_values_v) ** 2))
                errors_v.append(error_v)

        computation_time = time.time() - start_time

        # Return errors, computation time, and solutions if time_points are provided
        if time_points is not None:
            return errors_v, computation_time, solutions
        else:
            return errors_v, computation_time

    def solve_ode(self, t_start, t_end, v_values, w_values):
        num_nodes = len(v_values)

        def ode_system_vectorized(t, y_flat):
            """
            Vectorized ODE system for v and w.

            Parameters:
                t (float): Current time.
                y_flat (np.ndarray): Flattened array containing [v, w], shape (2 * num_nodes,)

            Returns:
                np.ndarray: Flattened derivatives [dv_dt, dw_dt], shape (2 * num_nodes,)
            """
            return self.ode_system(t, y_flat)

        # Initial conditions for ODE: [v, w]
        y0 = np.concatenate([v_values, w_values])

        # Solve ODE
        sol = solve_ivp(
            ode_system_vectorized,
            [t_start, t_end],
            y0,
            method='RK45',
            vectorized=False,
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Extract final values
        y_end = sol.y[:, -1]
        v_new = y_end[:num_nodes]
        w_new = y_end[num_nodes:]

        return v_new, w_new

    
    def setup_rhs_pde(self):
        dx = ufl.Measure("dx", domain=self.mesh)
        if self.ode_system is not None:
            # No explicit source term; RHS is just v_n
            self.L_v = self.v_n * self.v_v * dx
        else:
            # Use explicit source term
            x_coord, y_coord = ufl.SpatialCoordinate(self.mesh)
            I_app = self.source_term_func(x_coord, y_coord, self.time)
            self.L_v = (self.v_n + self.dt * I_app) * self.v_v * dx
                
        self.linear_form_v = fem.form(self.L_v)
        with self.b_v.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_v, self.linear_form_v)
    
    def solve_pde(self):
        self.solver_v.solve(self.b_v, self.v_h.vector)
        self.v_h.x.scatter_forward()


class BidomainSolverFEM(FEMBase):
    """
    FEM solver for the bidomain model.

    Attributes:
        M_i (fem.Constant or ufl.Expr): Intracellular conductivity tensor.
        M_e (fem.Constant or ufl.Expr): Extracellular conductivity tensor.
    """
    def __init__(self, N_x, N_y, T, stimulus_expression, M_i, M_e, dt, initial_v=0.0, initial_u_e=0.0):
        """
        Initialize the bidomain solver.

        Parameters:
            N_x (int): Number of elements in the x-direction.
            N_y (int): Number of elements in the y-direction.
            T (float): Total time for the simulation.
            stimulus_expression (ufl.Expr): Expression for the stimulus term.
            M_i (float or ufl.Expr): Intracellular conductivity tensor.
            M_e (float or ufl.Expr): Extracellular conductivity tensor.
            dt (float): Time step size.
            initial_v (float): Initial value for the solution.
            initial_u_e (float): Initial value for the extracellular potential.
        """
        super().__init__(N_x, N_y, T, dt, stimulus_expression, initial_v, initial_u_e)

        if isinstance(M_i, (int, float)):
            self.M_i = fem.Constant(self.domain, PETSc.ScalarType(M_i))
        else:
            self.M_i = ufl.as_tensor(M_i)

        if isinstance(M_e, (int, float)):
            self.M_e = fem.Constant(self.domain, PETSc.ScalarType(M_e))
        else:
            self.M_e = ufl.as_tensor(M_e)

        self.create_variational_problems()
        self.create_solvers()

    def create_variational_problems(self):
        """
        Create variational problems for the bidomain model.
        """
        # Define trial and test functions
        self.u_v, self.v_v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.u_u_e, self.v_u_e = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        # Define the source term
        x, y = ufl.SpatialCoordinate(self.domain)
        self.I_app = self.stimulus_expression(x, y, 0)

        # Define measures
        dx = ufl.Measure("dx", domain=self.domain)

        # Define variational problem for v
        self.a_v = (self.u_v * self.v_v + self.dt * ufl.dot(self.M_i * ufl.grad(self.u_v), ufl.grad(self.v_v))) * dx
        self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * dx

        # Define variational problem for u_e
        self.a_u_e = (self.u_u_e * self.v_u_e + self.dt * ufl.dot((self.M_i + self.M_e) * ufl.grad(self.u_u_e), ufl.grad(self.v_u_e))) * dx
        self.L_u_e = (self.u_e_n - self.dt * self.I_app) * self.v_u_e * dx

        # Assemble forms
        self.bilinear_form_v = fem.form(self.a_v)
        self.linear_form_v = fem.form(self.L_v)
        self.bilinear_form_u_e = fem.form(self.a_u_e)
        self.linear_form_u_e = fem.form(self.L_u_e)

        # Assemble matrices
        self.A_v = fem.petsc.assemble_matrix(self.bilinear_form_v)
        self.A_v.assemble()
        self.A_u_e = fem.petsc.assemble_matrix(self.bilinear_form_u_e)
        self.A_u_e.assemble()

        # Create vectors
        self.b_v = fem.petsc.create_vector(self.linear_form_v)
        self.b_u_e = fem.petsc.create_vector(self.linear_form_u_e)

    def create_solvers(self):
        """
        Create solvers for the bidomain model.
        """
        # Create solvers for v and u_e
        self.solver_v = PETSc.KSP().create(self.domain.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)

        self.solver_u_e = PETSc.KSP().create(self.domain.comm)
        self.solver_u_e.setOperators(self.A_u_e)
        self.solver_u_e.setType(PETSc.KSP.Type.PREONLY)
        self.solver_u_e.getPC().setType(PETSc.PC.Type.LU)

    def run(self, analytical_solution_v=None, analytical_solution_u_e=None, time_points=None):
        """
        Run the bidomain simulation.

        Parameters:
            analytical_solution_v (callable, optional): Analytical solution for validation of v.
            analytical_solution_u_e (callable, optional): Analytical solution for validation of u_e.
            time_points (list of float, optional): Time points to store the solution.

        Returns:
            tuple: Errors for v (list of float), errors for u_e (list of float), computation time (float), and optionally solutions at specified time points.
        """
        errors_v = []
        errors_u_e = []
        start_time = time.time()
        dof_coords = self.V.tabulate_dof_coordinates()
        x_coords = dof_coords[:, 0]
        y_coords = dof_coords[:, 1]

        # Prepare dictionary to store solutions at specified time points
        if time_points is not None:
            solutions_v = {t: None for t in time_points}
            solutions_u_e = {t: None for t in time_points}

        for i in range(self.num_steps + 1):
            t = i * self.dt  # Correctly set t for the current step

            # Update source term
            x, y = ufl.SpatialCoordinate(self.domain)
            self.I_app = self.stimulus_expression(x, y, t)
            
            # Update variational problems for v
            self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * ufl.dx
            self.linear_form_v = fem.form(self.L_v)
            with self.b_v.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b_v, self.linear_form_v)
            self.solver_v.solve(self.b_v, self.v_h.vector)
            self.v_h.x.scatter_forward()
            self.v_n.x.array[:] = self.v_h.x.array

            # Update variational problems for u_e
            self.L_u_e = (self.u_e_n - self.dt * self.I_app) * self.v_u_e * ufl.dx
            self.linear_form_u_e = fem.form(self.L_u_e)
            with self.b_u_e.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b_u_e, self.linear_form_u_e)
            self.solver_u_e.solve(self.b_u_e, self.u_e_h.vector)
            self.u_e_h.x.scatter_forward()
            self.u_e_n.x.array[:] = self.u_e_h.x.array

            # Store the solution if the current time is one of the specified time points
            if time_points is not None:
                for tp in time_points:
                    if np.isclose(t, tp, atol=1e-5):  # Increased tolerance
                        solutions_v[tp] = self.v_h.x.array.copy()
                        solutions_u_e[tp] = self.u_e_h.x.array.copy()

            # Calculate errors for v
            if analytical_solution_v is not None:
                analytical_values_v = analytical_solution_v(x_coords, y_coords, t)
                numerical_values_v = self.v_h.x.array
                error_v = np.sqrt(np.sum((numerical_values_v - analytical_values_v) ** 2) / len(analytical_values_v))
                errors_v.append(error_v)

            # Calculate errors for u_e
            if analytical_solution_u_e is not None:
                analytical_values_u_e = analytical_solution_u_e(x_coords, y_coords, t)
                numerical_values_u_e = self.u_e_h.x.array
                error_u_e = np.linalg.norm(numerical_values_u_e - analytical_values_u_e) / np.linalg.norm(analytical_values_u_e)
                errors_u_e.append(error_u_e)

        computation_time = time.time() - start_time

        # Return errors and solutions
        if time_points is not None:
            return errors_v, errors_u_e, computation_time, solutions_v, solutions_u_e
        else:
            return errors_v, errors_u_e, computation_time