import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import pyvista
import time

"""
This code uses FEM to solve the monodomain model in 2D meshgrid. Spesifically, the code seeks to reproduce the results from
https://finsberg.github.io/fenics-beat/tests/README.html to check the credability of the solver. The equation we solve 
reduces to

dv/dt = ∇²v + I_app which is essentially a diffusion equation with a source term.

Currently, the there are numerical unstabilites in the code which needs to be solved.

"""

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
dt = 0.01
num_steps = int(T / dt)

# Define mesh
nx, ny = 10, 10
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition for v as a constant function (0.0)
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(lambda x: np.zeros_like(x[0]))

# Define source term Istim
def stimulus_expression(t):
    x = ufl.SpatialCoordinate(domain)
    return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

# Define solution variable, and interpolate initial solution for visualization
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(lambda x: np.zeros_like(x[0]))

# Define analytical solution variable for visualization
analytical_function = fem.Function(V)
analytical_function.name = "analytical"

# Write initial condition to file
xdmf = io.XDMFFile(domain.comm, "paraview_files/monodomain_solution.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(uh, t)

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = stimulus_expression(0)  # Initial source term
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble matrix
A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

# Create linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Define analytical solution for comparison
def analytical_solution(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

errors = []

# Start timer
start_time = time.time()

# Time-stepping
for i in range(num_steps):
    t += dt

    # Update the source term
    f = stimulus_expression(t)

    # Update the right hand side form with the updated source term
    L = (u_n + dt * f) * v * ufl.dx
    linear_form = fem.form(L)

    # Update the right hand side vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Update and write analytical solution
    analytical_values = analytical_solution(domain.geometry.x[:, 0], domain.geometry.x[:, 1], t)
    analytical_function.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(t))

    # Compare with analytical solution
    numerical_values = uh.x.array
    error = np.linalg.norm(numerical_values - analytical_values)
    errors.append(error)

# Stop timer
end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

# Write final solution to file
xdmf.write_function(uh, t)
xdmf.write_function(analytical_function, t)
xdmf.close()

# Plot numerical and analytical solutions at final time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
X, Y = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

# Extract the data for plotting
numerical_solution = uh.x.array.reshape((ny+1, nx+1))
analytical_solution_values = analytical_solution(X, Y, T).reshape((ny+1, nx+1))

# Plot numerical solution
ax = axes[0]
contour = ax.contourf(X, Y, numerical_solution, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Numerical Solution at t={T:.5f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot analytical solution
ax = axes[1]
contour = ax.contourf(X, Y, analytical_solution_values, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Analytical Solution at t={T:.5f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle(f"Comparison of Numerical and Analytical Solutions of the Monodomain Model for dt = {dt:.1e}", fontweight='bold')
plt.savefig(f"figures/comparison_at_dt={dt:.1e}.pdf")
plt.show()

# Plot the error over time
plt.figure()
plt.plot(np.arange(num_steps) * dt, errors)
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title(f"Error over time for dt = {dt:.1e}")
plt.savefig(f"figures/error_for_dt={dt:.1e}.pdf")
plt.show()