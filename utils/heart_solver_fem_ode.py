import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
import os

class FEMBase:
    """
    Base class for finite element method (FEM) solvers.

    Attributes:
        N_x (int): Number of elements in the x-direction.
        N_y (int): Number of elements in the y-direction.
        T (float): Total time for the simulation.
        dt (float): Time step size.
        num_steps (int): Number of time steps.
        stimulus_expression (ufl.Expr): Expression for the stimulus term.
        domain (mesh.Mesh): Computational domain.
        V (fem.FunctionSpace): Function space for the problem.
        v_n (fem.Function): Solution at the previous time step.
        v_h (fem.Function): Solution at the current time step.
        u_e_n (fem.Function): Extracellular potential at the previous time step.
        u_e_h (fem.Function): Extracellular potential at the current time step.
    """

    def __init__(self, N_x, N_y, T, dt, stimulus_expression, initial_v=0.0, initial_u_e=0.0):
        self.N_x = N_x 
        self.N_y = N_y
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)
        self.stimulus_expression = stimulus_expression
        
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, N_x, N_y)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

        self.v_n = fem.Function(self.V)
        self.v_n.name = "v_n"
        self.v_n.interpolate(lambda x: np.zeros_like(x[0]))

        self.v_h = fem.Function(self.V)
        self.v_h.name = "v_h"
        self.v_h.interpolate(lambda x: np.zeros_like(x[0]))

        self.u_e_n = fem.Function(self.V)
        self.u_e_n.name = "u_e_n"
        self.u_e_n.interpolate(lambda x: np.zeros_like(x[0]))

        self.u_e_h = fem.Function(self.V)
        self.u_e_h.name = "u_e_h"
        self.u_e_h.interpolate(lambda x: np.zeros_like(x[0]))

class MonodomainSolverFEM(FEMBase):
    """
    FEM solver for the monodomain model.

    Attributes:
        M_i (fem.Constant or ufl.Expr): Conductivity tensor.
    """
    def __init__(self, N_x, N_y, T, stimulus_expression, M_i, dt, initial_v=0.0):
        """
        Initialize the monodomain solver.

        Parameters:
            N_x (int): Number of elements in the x-direction.
            N_y (int): Number of elements in the y-direction.
            T (float): Total time for the simulation.
            stimulus_expression (ufl.Expr): Expression for the stimulus term.
            M_i (float or ufl.Expr): Conductivity tensor.
            dt (float): Time step size.
            initial_v (float): Initial value for the solution.
        """
        if isinstance(M_i, (int, float)):
            self.M_i = fem.Constant(mesh.create_unit_square(MPI.COMM_WORLD, N_x, N_y), PETSc.ScalarType(M_i))
        else:
            self.M_i = ufl.as_tensor(M_i)

        super().__init__(N_x, N_y, T, dt, stimulus_expression, initial_v)

        self.create_variational_problems()
        self.create_solvers()

    def create_variational_problems(self):
        """
        Create variational problems for the monodomain model.
        """
        self.u_v, self.v_v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        x, y = ufl.SpatialCoordinate(self.domain)
        self.I_app = self.stimulus_expression(x, y, 0)  # Initial source term
        dx = ufl.Measure("dx", domain=self.domain)  # Specify the domain for integration
        self.a_v = (self.u_v * self.v_v + self.dt * ufl.dot(self.M_i * ufl.grad(self.u_v), ufl.grad(self.v_v))) * dx
        self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * dx

        self.bilinear_form_v = fem.form(self.a_v)
        self.linear_form_v = fem.form(self.L_v)

        self.A_v = fem.petsc.assemble_matrix(self.bilinear_form_v)
        self.A_v.assemble()

        self.b_v = fem.petsc.create_vector(self.linear_form_v)

    def create_solvers(self):
        """
        Create solvers for the monodomain model.
        """
        self.solver_v = PETSc.KSP().create(self.domain.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)

    def run(self, analytical_solution_v=None, time_points=None):
        """
        Run the monodomain simulation.

        Parameters:
            analytical_solution_v (callable, optional): Analytical solution for validation.
            time_points (list of float, optional): Time points to store the solution.

        Returns:
            tuple: Errors (list of float), computation time (float), and optionally solutions at specified time points.
        """
        errors_v = []
        start_time = time.time()
        dof_coords = self.V.tabulate_dof_coordinates()
        x_coords = dof_coords[:, 0]
        y_coords = dof_coords[:, 1]

        # Prepare dictionary to store solutions at specified time points
        if time_points is not None:
            solutions = {t: None for t in time_points}

        for i in range(self.num_steps + 1):
            t = i * self.dt  # Correctly set t for the current step

            x, y = ufl.SpatialCoordinate(self.domain)
            self.I_app = self.stimulus_expression(x, y, t)
            self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * ufl.dx
            self.linear_form_v = fem.form(self.L_v)

            with self.b_v.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b_v, self.linear_form_v)

            self.solver_v.solve(self.b_v, self.v_h.vector)
            self.v_h.x.scatter_forward()

            self.v_n.x.array[:] = self.v_h.x.array

            # Store the solution if the current time is one of the specified time points
            if time_points is not None:
                for tp in time_points:
                    if np.isclose(t, tp, atol=1e-5):  # Increased tolerance
                        solutions[tp] = self.v_h.x.array.copy()

            if analytical_solution_v is not None:
                analytical_values_v = analytical_solution_v(x_coords, y_coords, t)
                numerical_values_v = self.v_h.x.array
                error_v = 1/(len(analytical_values_v))*np.linalg.norm(numerical_values_v - analytical_values_v)
                errors_v.append(error_v)

        computation_time = time.time() - start_time

        # Return errors and solutions
        if time_points is not None:
            return errors_v, computation_time, solutions
        else:
            return errors_v, computation_time
        