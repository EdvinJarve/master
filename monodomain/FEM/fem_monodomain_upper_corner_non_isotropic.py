import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import time

"""
This code uses FEM to solve the 2D monodomain model. It handles a corner stimulation from t = 0 to t = 0.2 
with an amplitude of 200. The conductivity tensor is set to be linear, making the current downward linearly.y"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io
from petsc4py import PETSc
import time

t = 0  # Start time
T = 1.0  # Final time
num_steps = 20
dt = T/num_steps

# Define mesh
nx, ny = 20, 20
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition for v as a constant function (0.0)
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(lambda x: np.zeros_like(x[0]))
def stimulus_expression(t, duration=0.2, current_value = 50 ) :
    x = ufl.SpatialCoordinate(domain)
    
    # Define the spatial mask for the upper left corner
    spatial_mask = ufl.And(x[0] <= 0.2, x[1] >= 0.8)
    
    # Check the temporal condition
    if t <= duration:
        temporal_condition = current_value
    else:
        temporal_condition = 0.0
    
    # Apply the current value where the spatial condition is met
    return ufl.conditional(spatial_mask, temporal_condition, 0.0)



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
xdmf.write_function(uh, 0.0)

# Define the rotation matrix for 45 degrees
theta = np.pi / 4
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
R = ufl.as_matrix([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

# Define the diagonal conductivity matrix
D = ufl.as_matrix([[1, -1], [-1, 1]])

# Define non-isotropic conductivity tensor M
M = ufl.as_matrix([[1, -1], [-1, 1]])

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
t = 0.0  # Start time
f = stimulus_expression(t)  # Initial source term

# Variational formulation with non-isotropic conductivity tensor
a = (u * v + dt * ufl.dot(ufl.grad(u), M * ufl.grad(v))) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble matrix
A = fem.petsc.assemble_matrix(bilinear_form)
A.assemble()
b = fem.petsc.create_vector(linear_form)

# Create linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


# Start timer
start_time = time.time()
dof_coords = V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

# Store solutions for specified time points
time_points = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0]
solutions = {t: None for t in time_points}

# Time-stepping
for i in range(num_steps+1):
    # Current time
    t = i * dt

    # Update the right-hand side form with the updated source term
    f = stimulus_expression(t)
    L = (u_n + dt * f) * v * ufl.dx
    linear_form = fem.form(L)

    # Update the right-hand side vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear_form)

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at the previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Store the solution if the current time is one of the specified time points
    for tp in time_points:
        if np.isclose(t, tp, atol=1e-5):  # Increased tolerance
            solutions[tp] = uh.x.array.copy()
            #print(f"Solution at t={tp}:\n{uh.x.array}")

# Stop timer
end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

# Pairwise time points for plotting
pairwise_time_points = [(0.0, 0.1), (0.2, 0.3), (0.6, 1.0)]

# Prepare for plotting
for pair in pairwise_time_points:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    for idx, t_eval in enumerate(pair):
        numerical_solution = solutions[t_eval]  # Get the numerical solution at the specific time point
        
        # Ensure proper shape and check if solution exists
        if numerical_solution is not None:
            numerical_solution = numerical_solution[:len(x_coords)]
        
            # Plot numerical solution
            ax = axes[idx]
            contour = ax.tricontourf(x_coords, y_coords, numerical_solution, levels=50, cmap='viridis')
            fig.colorbar(contour, ax=ax)
            ax.set_title(f'Numerical Solution at t={t_eval}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            ax.set_title(f'No solution at t={t_eval}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_ylim(0, 1)  # Set y-axis range to be consistent

    # Set the main title
    #fig.suptitle(f"Numerical Solutions of the Monodomain Model with Non-Isotropic Conductivity Tensor", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"figures_fem_mono/FEM_solution_for_non_iso_corner_current_t{pair[0]}_t{pair[1]}.png")
    plt.show()

# Save the final numerical solution to file
xdmf.write_function(uh, T)
xdmf.close()

"""
Computation time (excluding visualization): 0.28 seconds

"""