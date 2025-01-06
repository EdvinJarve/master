import sys
import os
import numpy as np
import time
from dolfinx import fem, mesh
from mpi4py import MPI
import torch
import torch.nn as nn
import ufl
from matplotlib.ticker import ScalarFormatter
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata  # Ensure griddata is imported

# **Set the Matplotlib backend to 'Agg' before importing pyplot**
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import matplotlib.pyplot as plt  # Now safe to import pyplot

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

# Add the project root directory to sys.pathmo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs  # Ensure this class handles data loss properly

# Define the results directory relative to the project root
results_dir = os.path.join(project_root, 'monodomain_results', 'inverse_problem_analytical')
os.makedirs(results_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# =============================================================================
# Define the Source Term and Analytical Solution
# =============================================================================

# Define the source term function as provided
def source_term_func_pinns(x_spatial, t):
    """
    Source term function for PINNs.

    Args:
        x_spatial (torch.Tensor): Spatial coordinates (x, y).
        t (torch.Tensor): Time coordinate.

    Returns:
        torch.Tensor: Source term values.
    """
    pi = torch.pi
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    return (
        8 * pi**2 * torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.sin(t)
        + torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.cos(t)
    )

# Define the analytical solution
def analytical_solution(x, y, t):
    """
    Analytical solution for the monodomain equation.

    Args:
        x (numpy.ndarray): Spatial coordinate x.
        y (numpy.ndarray): Spatial coordinate y.
        t (numpy.ndarray): Time coordinate.

    Returns:
        numpy.ndarray: Solution v at given (x, y, t).
    """
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

# Device configuration
device = 'cpu'
print(f"Using device: {device}")

# =============================================================================
# Initialize the Inverse PINN Solver
# =============================================================================

# Initialize the inverse PINN solver
pinn = InverseMonodomainSolverPINNs(
    num_inputs=3,  # For x, y, t
    num_layers=3,
    num_neurons=256,
    device=device,
    source_term_func=source_term_func_pinns,
    initial_M=1e-2,  # Starting guess for M
    loss_weights={
        'pde_loss': 1.0,
        'IC_loss': 1.0,
        'BC_loss': 1.0,
        'data_loss': 3.0,
        'ode_loss': 0.0  # Set to 0 if not using ODE
    }
)

# Define your optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)

# =============================================================================
# Data Generation
# =============================================================================

# Define the spatial and temporal domains
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Number of points
num_ic = 4000
num_collocation = 10000
num_boundary = 4000
num_data = 10000  # Number of data points within the domain
true_M = 1.0

# Initial Condition Data (t = t_min)
X_ic = np.random.uniform([x_min, y_min], [x_max, y_max], (num_ic, 2))
t_ic = np.full((num_ic, 1), t_min)
X_ic_full = np.hstack((X_ic, t_ic))
expected_u0 = analytical_solution(X_ic[:, 0], X_ic[:, 1], t_ic[:, 0])

# **Data Points within the Domain (Newly Added)**
# Generate random points within the domain
X_data = np.random.uniform([x_min, y_min, t_min], [x_max, y_max, t_max], (num_data, 3))
expected_data = analytical_solution(X_data[:, 0], X_data[:, 1], X_data[:, 2])

# Collocation Points
X_collocation = np.random.uniform([x_min, y_min, t_min], [x_max, y_max, t_max], (num_collocation, 3))

# Boundary Condition Data (Neumann BCs)
def generate_boundary_points(x_min, x_max, y_min, y_max, t_min, t_max, num_points):
    # Generate points on the boundary of the spatial domain at random times
    X_boundary = []
    for _ in range(num_points):
        t = np.random.uniform(t_min, t_max)
        side = np.random.choice(['x_min', 'x_max', 'y_min', 'y_max'])
        if side == 'x_min':
            x = x_min
            y = np.random.uniform(y_min, y_max)
        elif side == 'x_max':
            x = x_max
            y = np.random.uniform(y_min, y_max)
        elif side == 'y_min':
            x = np.random.uniform(x_min, x_max)
            y = y_min
        else:  # y_max
            x = np.random.uniform(x_min, x_max)
            y = y_max
        X_boundary.append([x, y, t])
    return np.array(X_boundary)

def compute_normal_vectors(X_boundary):
    # Compute normal vectors for Neumann boundary conditions
    normals = []
    for point in X_boundary:
        x, y, _ = point
        if np.isclose(x, x_min):
            normals.append([-1, 0])  # Left boundary
        elif np.isclose(x, x_max):
            normals.append([1, 0])   # Right boundary
        elif np.isclose(y, y_min):
            normals.append([0, -1])  # Bottom boundary
        elif np.isclose(y, y_max):
            normals.append([0, 1])   # Top boundary
        else:
            normals.append([0, 0])   # Interior point (should not happen)
    return np.array(normals)

X_boundary = generate_boundary_points(x_min, x_max, y_min, y_max, t_min, t_max, num_boundary)
normal_vectors = compute_normal_vectors(X_boundary)

# =============================================================================
# Convert Data to Tensors and Assign to Model
# =============================================================================

# Convert initial condition data to tensors
pinn.X_ic = torch.tensor(X_ic_full, dtype=torch.float32).to(device)
pinn.expected_u0 = torch.tensor(expected_u0, dtype=torch.float32).unsqueeze(-1).to(device)

# Convert collocation points to tensors
pinn.X_collocation = torch.tensor(X_collocation, dtype=torch.float32).to(device)

# Convert boundary condition data to tensors
pinn.X_boundary = torch.tensor(X_boundary, dtype=torch.float32).to(device)
pinn.normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32).to(device)

# **Convert domain data points to tensors (Newly Added)**
pinn.X_data = torch.tensor(X_data, dtype=torch.float32).to(device)
pinn.expected_data = torch.tensor(expected_data, dtype=torch.float32).unsqueeze(-1).to(device)

# =============================================================================
# Training Loop with Loss and M Tracking
# =============================================================================

# Initialize lists to store loss and M values
total_loss_list = []
M_estimates = []
pde_loss_list = []
IC_loss_list = []
BC_loss_list = []
data_loss_list = []

num_epochs = 30000
for epoch in range(num_epochs):
    pde_loss, IC_loss, BC_loss, data_loss, ode_loss, total_loss = pinn.train_step(optimizer, batch_size=126)
    
    # Record losses and M estimate
    total_loss_list.append(total_loss)
    M_estimates.append(pinn.M.item())
    pde_loss_list.append(pde_loss)
    IC_loss_list.append(IC_loss)
    BC_loss_list.append(BC_loss)
    data_loss_list.append(data_loss)
    
    # Every 100 epochs, print the losses and current estimate of M
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss:.6f}, PDE Loss = {pde_loss:.6f}, IC Loss = {IC_loss:.6f}, BC Loss = {BC_loss:.6f}, Data Loss = {data_loss:.6f}, M Estimate = {pinn.M.item():.6f}")

# =============================================================================
# Plotting and Saving Loss and M Estimates
# =============================================================================

# Plot Total Loss over Epochs
plt.figure(figsize=(10, 5))
plt.plot(total_loss_list, label='Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Total Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'total_loss_over_epochs.png'))
plt.close()

# Plot M Estimates over Epochs
plt.figure(figsize=(10, 5))
plt.plot(M_estimates, label='Estimated M', color='orange')
plt.axhline(y=true_M, color='red', linestyle='--', label='True M')
plt.xlabel('Epochs')
plt.ylabel('M Value')
plt.title('Estimated M Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'M_estimates_over_epochs.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(pde_loss_list, label='PDE Loss')
plt.plot(data_loss_list, label='Data Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('PDE and Data Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'pde_data_loss_over_epochs.png'))
plt.close()
