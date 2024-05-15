import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl
import pyvista
import matplotlib.pyplot as plt 
import time

"""
Code that uses FenicsX to solve the monodomain equation for a 2D meshgrid. Specifically, we check if the code 
can reproduse analytical results made by Henrik Finsberg: https://finsberg.github.io/fenics-beat/tests/README.html.

The equation reduces to 

dv/dt = ∇²v + I_app, which is the diffusion equation with a source term.

As of now, the code the contourplots indicate numerical unstability. However, the GIFs seems to show that the numerical 
behaviour is very similar to the behaviour of the analytical solution. 

"""

# Define temporal parameters
t = 0  # Start time
T = 0.1  # Final time
dt = 1e-3
num_steps = int(T / dt)

# Define equation variables, not needed in the current code
M = 1
chi = 1 

# Define mesh
nx, ny = 10, 10
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(lambda x: np.zeros_like(x[0]))

# Define source term
def stimulus_expression(t):
    x = ufl.SpatialCoordinate(domain)
    return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

# Define solution variable
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

# No boundary conditions is given as the boundary condition naturally vanishes using FEM

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

# Visualization commented out to speed up the calculations
"""
# Visualization setup
pyvista.start_xvfb()
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

# Analytical solution and plotting 
grid_analytical = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter_analytical = pyvista.Plotter()
plotter_analytical.open_gif("(gifs/analytical_time.gif", fps=10)

grid.point_data["uh"] = uh.x.array
grid_analytical.point_data["analytical"] = analytical_function.x.array

warped = grid.warp_by_scalar("uh", factor=1)
warped_analytical = grid_analytical.warp_by_scalar("analytical", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

plotter.add_mesh(warped, show_edges=True, lighting=False,
                 cmap=viridis, scalar_bar_args=sargs)

plotter_analytical.add_mesh(warped_analytical, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs)

"""

# Define analytical solution for comparison
def analytical_solution(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

errors = []

# Time-stepping
start_time = time.time()

for i in range(num_steps):

    t += dt

    # Update source term
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

    # Write solution to file
    xdmf.write_function(uh, t)

    # Update and write analytical solution
    analytical_values = analytical_solution(domain.geometry.x[:, 0], domain.geometry.x[:, 1], t)
    analytical_function.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(t))
    xdmf.write_function(analytical_function, t)

    # Compare with analytical solution
    numerical_values = uh.x.array
    error = np.linalg.norm(numerical_values - analytical_values)
    errors.append(error)

# Timer end
end_time = time.time()
computation_time = end_time - start_time

# Print computation time
print(f"Computation Time: {computation_time:.5f} seconds")

# Finalize XDMF file
xdmf.close()

# Plot numerical and analytical solutions at final time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
X, Y = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

# Plot numerical solution
ax = axes[0]
numerical_solution = uh.x.array.reshape((ny+1, nx+1))
contour = ax.contourf(X, Y, numerical_solution, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Numerical Solution at t={T:.5f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot analytical solution
ax = axes[1]
analytical_solution_values = analytical_solution(X, Y, T)
contour = ax.contourf(X, Y, analytical_solution_values, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Analytical Solution at t={T:.3f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle(f"Comparison of FEM and Analytical Solutions of the Monodomain Model for dt = {dt:.1e}", fontweight='bold')
plt.savefig(f"figures/fem_comparison_at_dt={dt:.3f}_at_T={T:.3f}.pdf")
plt.close()

# Plot the error over time
plt.figure()
plt.plot(np.arange(num_steps) * dt, errors)
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title(f"Error over time for dt = {dt:.1e} using FEM")
plt.savefig(f"figures/fem_error_for_dt={dt:.1e}.pdf")
plt.close()