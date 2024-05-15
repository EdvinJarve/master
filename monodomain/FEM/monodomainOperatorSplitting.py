import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import Function
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from ufl import TrialFunction, TestFunction, dx, grad, dot
import matplotlib.pyplot as plt

""""
Class the solves the monodomain model using operator splitting.

As of now, the class is hardcoded to try to
recreate the results from a known analytical solution made by Henrik Finsberg.

The results can be found here: https://finsberg.github.io/fenics-beat/tests/README.html

Current status: The class does reproduce analytical results, as we experience big numerical unstability.
The same numerical issue seems to appear in fenics_monodomain.py as well

"""
class MonodomainSolverOperatorSplitting:
    def __init__(self, mesh_type, mesh_args, element_degree, T, dt):
        self.comm = MPI.COMM_WORLD
        if mesh_type == "unit_square":
            self.domain = mesh.create_unit_square(self.comm, *mesh_args)
        elif mesh_type == "unit_cube":
            self.domain = mesh.create_unit_cube(self.comm, *mesh_args)
        else:
            raise ValueError("Unsupported mesh type")
        
        self.V = fem.functionspace(self.domain, ("Lagrange", element_degree))
        
        self.mesh_args = mesh_args
        self.T = T
        self.dt = dt
        self.steps = int(T / dt)
        self.errors = []

    # Define the system of ODEs
    def system_of_odes(self, t, y):
        v, s = y
        dvdt = -s
        dsdt = v
        return [dvdt, dsdt]
    
    # Function for the stimulus
    def stimulus_expression(self, t, domain):
        x = ufl.SpatialCoordinate(domain)
        return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

    def solve_ODE_RK2(self, v0_flat, s0_flat, t_start, dt):
        x_points, y_points = self.mesh_args
        num_points = (x_points + 1) * (y_points + 1)
        v_flat = np.zeros(num_points)
        s_flat = np.zeros(num_points)

        for idx in range(num_points):
            v = v0_flat[idx]
            s = s0_flat[idx]
            dvdt1, dsdt1 = self.system_of_odes(t_start, [v, s])
            v1 = v + dvdt1 * dt * 0.5  # Midpoint
            s1 = s + dsdt1 * dt * 0.5  # Midpoint
            dvdt2, dsdt2 = self.system_of_odes(t_start + dt * 0.5, [v1, s1])
            v_flat[idx] = v + dvdt2 * dt
            s_flat[idx] = s + dsdt2 * dt

        return v_flat, s_flat

    def solve_PDE(self, t_start, t_end, v_init):
        v_n = fem.Function(self.V)
        v_n.vector.array[:] = v_init

        uh = fem.Function(self.V)
        uh.name = "uh"

        t = fem.Constant(self.domain, PETSc.ScalarType(t_start))
        stimulus = self.stimulus_expression(t, self.domain)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = u * v * ufl.dx + self.dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (v_n + self.dt * stimulus) * v * ufl.dx

        bilinear_form = fem.form(a)
        linear_form = fem.form(L)

        A = fem.petsc.assemble_matrix(bilinear_form)
        A.assemble()
        b = fem.petsc.create_vector(linear_form)

        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        fem.petsc.assemble_vector(b, linear_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return np.copy(uh.vector.array)

    def solve_monodomain(self, initial_v, initial_s, theta=0.5):
        i = 0
        v = np.copy(initial_v)
        s = np.copy(initial_s)

        while i < self.steps:
            # Printing for debugging 

            # First ODE step for theta*dt
            v_theta, s_theta = self.solve_ODE_RK2(v, s, i * self.dt, theta * self.dt)
            print(f"Step {i}: v_theta: {v_theta}, s_theta: {s_theta}")

            # Update v with PDE solver from t_n to t_n + dt
            uh_vector = self.solve_PDE(i * self.dt, (i + 1) * self.dt, v_theta)
            print(f"Step {i}: uh_vector (after PDE): {uh_vector}")

            # Second ODE step for (1 - theta)*dt
            v, s = self.solve_ODE_RK2(uh_vector, s_theta, (i + 1) * self.dt, (1 - theta) * self.dt)
            print(f"Step {i}: v (after second ODE): {v}, s: {s}")

            # Calculate error at each step
            error = self.calculate_error(v)
            self.errors.append(error)
            print(f"Step {i}: Error: {error}")

            i += 1

        return v

    def calculate_error(self, v_numerical):
        x_points = np.linspace(0, 1, self.mesh_args[0] + 1)
        y_points = np.linspace(0, 1, self.mesh_args[1] + 1)
        X, Y = np.meshgrid(x_points, y_points)
        T = self.steps * self.dt
        v_analytical = self.analytical_solution(X, Y, T).flatten()
        error = np.linalg.norm(v_numerical - v_analytical)
        return error

    def analytical_solution(self, x, y, t):
        return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)