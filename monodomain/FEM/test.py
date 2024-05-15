from monodomainOperatorSplitting import MonodomainSolverOperatorSplitting
import numpy as np
import matplotlib.pyplot as plt

# Define the mesh resolution
dt = 1e-6
T = 0.1
n = 10  # Define the resolution
x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)
X, Y = np.meshgrid(x, y)

# Calculate initial conditions over the meshgrid
v0 = np.zeros_like(X)
s0 = -np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

v0_flat = v0.flatten()
s0_flat = s0.flatten()

# Instantiate the solver
solver = MonodomainSolverOperatorSplitting(
    mesh_type="unit_square",
    mesh_args=(n, n),  # Simpler mesh for testing
    element_degree=1,
    T=T,
    dt=dt
)

# Generate mesh points for initial conditions directly from the Dolfinx mesh
V = solver.V
dof_coordinates = V.tabulate_dof_coordinates().reshape(-1, 3)
dof_coordinates = dof_coordinates[:, :2]

x = dof_coordinates[:, 0]
y = dof_coordinates[:, 1]

# Define simple initial conditions
v0 = np.zeros_like(x)
s0 = -np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

# Run the solver
g = solver.solve_monodomain(v0_flat, s0_flat)

# Compare solutions
error = solver.calculate_error(g)
print(f"Error between numerical and analytical solutions: {error}")

# Plot numerical and analytical solutions side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot numerical solution
ax = axes[0]
contour = ax.contourf(X, Y, g.reshape((n + 1, n + 1)), levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Numerical Solution at t={T:.5f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot analytical solution
ax = axes[1]
contour = ax.contourf(X, Y, solver.analytical_solution(X, Y, T).reshape((n + 1, n + 1)), levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'Analytical Solution at t={T:.5f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.suptitle(f"Comparison of Numerical and Analytical Solutions of the Monodomain Model for dt = {solver.dt:.1e}", fontweight='bold')
plt.savefig(f"figures/comparison_at_dt={solver.dt:.1e}.pdf")
plt.show()

# Plot the error over time
plt.figure()
plt.plot(np.arange(solver.steps) * dt, solver.errors)
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title(f"Error over time for dt = {solver.dt:.1e}")
plt.savefig(f"figures/error_for_dt={solver.dt:.1e}.pdf")
plt.show()