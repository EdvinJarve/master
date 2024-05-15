import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl
import pyvista
import matplotlib.pyplot as plt 

import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl
import pyvista

# Define temporal parameters
t = 0  # Start time
T = 0.2  # Final time
dt = 1e-3
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
xdmf = io.XDMFFile(domain.comm, "monodomain_solution.xdmf", "w")
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

# Visualization setup
pyvista.start_xvfb()
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

# Analytical solution visualization setup
grid_analytical = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter_analytical = pyvista.Plotter()
plotter_analytical.open_gif("analytical_time.gif", fps=10)

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

# Define analytical solution for comparison
def analytical_solution(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

errors = []

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

    # Debug output for b
    print(f"Time step {i}, b vector: {b.array[:5]}")

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Debug output for uh
    print(f"Time step {i}, uh vector: {uh.x.array[:5]}")

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

    # Update plots
    grid.point_data["uh"] = uh.x.array
    warped.points[:, :] = grid.warp_by_scalar("uh", factor=1).points
    plotter.update_coordinates(warped.points, render=True)
    plotter.update_scalars(uh.x.array, render=True)
    plotter.write_frame()

    grid_analytical.point_data["analytical"] = analytical_function.x.array
    warped_analytical.points[:, :] = grid_analytical.warp_by_scalar("analytical", factor=1).points
    plotter_analytical.update_coordinates(warped_analytical.points, render=True)
    plotter_analytical.update_scalars(analytical_function.x.array, render=True)
    plotter_analytical.write_frame()

# Finalize
plotter.close()
plotter_analytical.close()
xdmf.close()

# Debug output for errors
print(f"Errors: {errors}")


# Debug output for errors
print(f"Errors: {errors}")
print("g")
times = np.linspace(0,1, num_steps)

plt.title("Error as a function of time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.plot(times, errors)
plt.savefig("error.png")
plt.close()