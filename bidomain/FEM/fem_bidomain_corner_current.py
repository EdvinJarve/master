import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
import numpy as np
from petsc4py import PETSc
import time
import matplotlib.pyplot as plt
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, create_vector)
import dolfinx.jit
from dolfinx import default_scalar_type

# Define temporal parameters

# Define temporal parameters
T = 1.0  # Final time
num_steps = 40
dt = T / num_steps

# Define mesh
nx, ny = 40, 40
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(domain, ("Lagrange", 1))

v_n = fem.Function(V)
v_n.name = "v_n"
v_n.interpolate(lambda x: np.zeros_like(x[0]))

u_e_n = fem.Function(V)
u_e_n.name = "u_e_n"
u_e_n.interpolate(lambda x: np.zeros_like(x[0]))

# Define the non-isotropic conductivity tensors
M_i = fem.Constant(domain, PETSc.ScalarType(1.0))  # Example value for M_i
M_e = fem.Constant(domain, PETSc.ScalarType(1.0))  # Example value for M_e

def stimulus_expression(t, duration=0.2, current_value = 100):
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

# Define solution variables for v and u_e, and interpolate initial solutions for visualization
v_h = fem.Function(V)
v_h.name = "v_h"
v_h.interpolate(lambda x: np.zeros_like(x[0]))

u_e_h = fem.Function(V)
u_e_h.name = "u_e_h"
u_e_h.interpolate(lambda x: np.zeros_like(x[0]))

# Define variational problem for v
u_v, v_v = ufl.TrialFunction(V), ufl.TestFunction(V)
a_v = (u_v * v_v + dt * ufl.dot(M_i * ufl.grad(u_v), ufl.grad(v_v))) * ufl.dx

# Define variational problem for u_e
u_u_e, v_u_e = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u_e = (u_u_e * v_u_e + dt * ufl.dot((M_i + M_e) * ufl.grad(u_u_e), ufl.grad(v_u_e))) * ufl.dx

bilinear_form_v = fem.form(a_v)
bilinear_form_u_e = fem.form(a_u_e)

# Assemble matrices
A_v = fem.petsc.assemble_matrix(bilinear_form_v)
A_v.assemble()
A_u_e = fem.petsc.assemble_matrix(bilinear_form_u_e)
A_u_e.assemble()

b_v = fem.petsc.create_vector(bilinear_form_v)
b_u_e = fem.petsc.create_vector(bilinear_form_u_e)

# Create linear solvers
solver_v = PETSc.KSP().create(domain.comm)
solver_v.setOperators(A_v)
solver_v.setType(PETSc.KSP.Type.PREONLY)
solver_v.getPC().setType(PETSc.PC.Type.LU)

solver_u_e = PETSc.KSP().create(domain.comm)
solver_u_e.setOperators(A_u_e)
solver_u_e.setType(PETSc.KSP.Type.PREONLY)
solver_u_e.getPC().setType(PETSc.PC.Type.LU)


# Start timer
start_time = time.time()
dof_coords = V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

# Define time points for visualization
time_points = [0.0, 0.2, 0.4, 0.6, 1.0]
results_v = []
results_u_e = []


# Time-stepping
for i in range(num_steps + 1):
    t = i * dt
    
    # Update the source term
    I_app = stimulus_expression(t)

    # Update the right-hand side form with the updated source term for v
    L_v = (v_n + dt * I_app) * v_v * ufl.dx
    linear_form_v = fem.form(L_v)

    # Update the right-hand side vector for v
    with b_v.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b_v, linear_form_v)

    # Solve linear problem for v
    solver_v.solve(b_v, v_h.vector)
    v_h.x.scatter_forward()

    # Update solution at the previous time step (v_n)
    v_n.x.array[:] = v_h.x.array

    # Update the right-hand side form with the updated source term for u_e
    L_u_e = (u_e_n - dt * I_app) * v_u_e * ufl.dx
    linear_form_u_e = fem.form(L_u_e)

    # Update the right-hand side vector for u_e
    with b_u_e.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b_u_e, linear_form_u_e)

    # Solve linear problem for u_e
    solver_u_e.solve(b_u_e, u_e_h.vector)
    u_e_h.x.scatter_forward()

    # Update solution at the previous time step (u_e_n)
    u_e_n.x.array[:] = u_e_h.x.array

    # Store results for visualization at specified time points
    for time_point in time_points:
        if np.isclose(t, time_point, atol=dt / 2):
            print(f"Storing results for t = {t}")
            results_v.append(v_h.x.array.copy())
            results_u_e.append(u_e_h.x.array.copy())
            break


# Stop timer
end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

# Extract coordinates of degrees of freedom
dof_coords = V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

# Create a meshgrid for contour plotting
X, Y = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))

# Plot solutions and errors
for idx, time_point in enumerate(time_points):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))

    numerical_solution_v = results_v[idx]
    numerical_solution_u_e = results_u_e[idx]

    # Plot numerical solution for v
    ax = axes[0]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_v, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Numerical Solution for v at t={time_point}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot numerical solution for u_e
    ax = axes[1]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Numerical Solution for u_e at t={time_point}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(f"figures_bi_fem/bi_corner_results_at_{time_point}.pdf")
    plt.show()


"""
Results:

Computation time (excluding visualization): 0.71 seconds

"""