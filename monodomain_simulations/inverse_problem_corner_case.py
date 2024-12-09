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
import json

# **Set the Matplotlib backend to 'Agg' before importing pyplot**
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import matplotlib.pyplot as plt  # Now safe to import pyplot

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs  # Ensure this class handles data loss properly

# Define the results directory relative to the project root
results_dir = os.path.join(project_root, 'monodomain_results', 'inverse_problem_corner_case')
os.makedirs(results_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# =============================================================================
# Hyperparameters Definition
# =============================================================================

hyperparams = {
    'num_inputs': 3,             # Number of inputs to the PINN (x, y, t)
    'num_layers': 2,             # Number of hidden layers in the neural network
    'num_neurons': 256,          # Number of neurons per hidden layer
    'device': 'cpu',            # Device configuration
    'initial_M': 0.0,          # Initial guess for M
    'learning_rate': 1e-4,      # Learning rate for the optimizer
    'pde_epochs': 4000,         # Number of epochs for PDE-only training
    'total_epochs': 30000,      # Total number of training epochs
    'batch_size': 32,
    'loss_weights': {           # Weights for different loss components
        'pde_loss': 1.0,
        'IC_loss': 0.333,
        'BC_loss': 0.333,
        'data_loss': 1000.0,    # Increased weight for data loss
        'ode_loss': 0.0         # Set to 0 if not using ODE
    },
    'source_term_initial_M': 1.0,  # True value of M used in FEM simulation
    'num_ic': 500,
    'num_collocation': 5000,
    'num_boundary': 4000,
    'num_data': 5000            # Number of data points within the domain
}

# =============================================================================
# Source Term Function for FEM using UFL
# =============================================================================

def source_term_func(x, y, t):
    """
    Gaussian stimulus applied in the upper left corner between x ∈ [0,0.2] and y ∈ [0.8,1],
    smoothed with a Gaussian distribution. The stimulus is turned on from t=0.05 to t=0.2,
    remains constant until t=0.4, and turns off steadily until t=0.55.

    Parameters:
        x (ufl.Variable): Spatial coordinate x.
        y (ufl.Variable): Spatial coordinate y.
        t (fem.Constant): Current time.

    Returns:
        ufl.Expression: Source term expression.
    """
    x0 = 0.2  # Center of the Gaussian in x
    y0 = 0.8  # Center of the Gaussian in y
    sigma = 0.03  # Standard deviation for the Gaussian
    t_on_start = 0.05
    t_on_end = 0.2
    t_off_start = 0.4
    t_off_end = 0.55

    # Gaussian distribution
    gaussian = 50 * ufl.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Define time window with ramp-up, constant, and ramp-down phases
    time_window = ufl.conditional(
        ufl.And(t >= t_on_start, t < t_on_end),
        (t - t_on_start) / (t_on_end - t_on_start),
        ufl.conditional(
            ufl.And(t >= t_on_end, t < t_off_start),
            1.0,
            ufl.conditional(
                ufl.And(t >= t_off_start, t <= t_off_end),
                1 - (t - t_off_start) / (t_off_end - t_off_start),
                0.0
            )
        )
    )

    return gaussian * time_window

def source_term_func_pinns(x_spatial, t):
    """
    Gaussian stimulus applied in the upper left corner between x ∈ [0,0.2] and y ∈ [0.8,1],
    smoothed with a Gaussian distribution. The stimulus is turned on from t=0.05 to t=0.2,
    remains constant until t=0.4, and turns off steadily until t=0.55.

    Parameters:
        x_spatial (torch.Tensor): Spatial coordinates tensor of shape (N, 2).
        t (torch.Tensor): Time tensor of shape (N, 1).

    Returns:
        torch.Tensor: Source term tensor of shape (N, 1).
    """
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    x0 = 0.2  # Center of the Gaussian in x
    y0 = 0.8  # Center of the Gaussian in y
    sigma = 0.03  # Standard deviation for the Gaussian
    t_on_start = 0.05
    t_on_end = 0.2
    t_off_start = 0.4
    t_off_end = 0.55

    # Gaussian distribution
    gaussian = 50 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Define time window with ramp-up, constant, and ramp-down phases
    time_window = torch.zeros_like(t)
    
    # Ramp-up phase
    ramp_up = (t >= t_on_start) & (t < t_on_end)
    time_window[ramp_up] = (t[ramp_up] - t_on_start) / (t_on_end - t_on_start)
    
    # Constant phase
    constant = (t >= t_on_end) & (t < t_off_start)
    time_window[constant] = 1.0
    
    # Ramp-down phase
    ramp_down = (t >= t_off_start) & (t <= t_off_end)
    time_window[ramp_down] = 1 - (t[ramp_down] - t_off_start) / (t_off_end - t_off_start)
    
    return gaussian * time_window


# =============================================================================
# FEM Simulation or Data Loading
# =============================================================================

fem_data_filename = os.path.join(results_dir, 'fem_simulation_data.npz')

if os.path.exists(fem_data_filename):
    # Load the FEM data from the file
    print(f"Loading FEM simulation data from {fem_data_filename}")
    fem_data = np.load(fem_data_filename)
    x_coords = fem_data['x_coords']
    y_coords = fem_data['y_coords']
    time_points = fem_data['time_points']
    solutions_fem_array = fem_data['solutions_fem']
else:
    # Run FEM simulation to generate data
    print("FEM simulation data not found. Running FEM simulation...")
    # Define mesh parameters and temporal parameters
    Nx, Ny, Nt = 100, 100, 100  # Spatial and temporal resolution
    T = 1.0                  # Final time
    dt = T / Nt              # Time step size
    M = hyperparams['source_term_initial_M']  # Conductivity

    # Create mesh
    domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)

    # Initialize and run the FEM simulation with the MonodomainSolverFEM class
    sim_fem = MonodomainSolverFEM(
        mesh=domain_mesh,
        T=T,
        dt=dt,
        M_i=M,
        source_term_func=source_term_func,  # Updated function with 3 arguments
        initial_v=0.0  # Optional since default is 0.0
    )

    # Define time points to store solutions
    time_points = np.linspace(0, T, Nt + 1)  # All time steps

    # Run FEM simulation with time_points to obtain solutions_fem
    print("Starting FEM simulation...")
    start_time_fem = time.time()
    errors_fem, computation_time_fem, solutions_fem = sim_fem.run(time_points=time_points)
    end_time_fem = time.time()
    print(f"FEM simulation complete in {end_time_fem - start_time_fem:.2f} seconds\n")

    # Extract coordinates of degrees of freedom
    dof_coords = sim_fem.V.tabulate_dof_coordinates()
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]

    # Process solutions_fem into a numpy array
    if isinstance(solutions_fem, dict):
        # Convert dict to array
        solutions_fem_array = np.array([solutions_fem[t] for t in time_points])
    elif isinstance(solutions_fem, list):
        solutions_fem_array = np.array(solutions_fem)
    elif isinstance(solutions_fem, np.ndarray):
        solutions_fem_array = solutions_fem
    else:
        raise TypeError("solutions_fem has an unexpected type.")

    # Save the FEM data to a file
    np.savez_compressed(
        fem_data_filename,
        x_coords=x_coords,
        y_coords=y_coords,
        time_points=time_points,
        solutions_fem=solutions_fem_array
    )
    print(f"FEM simulation data saved to {fem_data_filename}")

# =============================================================================
# Data Generation for PINN Training
# =============================================================================

# Flatten the data for easier handling
num_points = len(x_coords)
num_times = len(time_points)
X_full = np.zeros((num_points * num_times, 3))
v_full = np.zeros((num_points * num_times))

# Assign solutions to v_full correctly based on the structure of solutions_fem_array
for i, t in enumerate(time_points):
    idx_start = i * num_points
    idx_end = idx_start + num_points
    X_full[idx_start:idx_end, 0] = x_coords
    X_full[idx_start:idx_end, 1] = y_coords
    X_full[idx_start:idx_end, 2] = t
    v_full[idx_start:idx_end] = solutions_fem_array[i]

# Define the spatial and temporal domains
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Number of points
num_ic = hyperparams.get('num_ic', 4000)
num_collocation = hyperparams.get('num_collocation', 10000)
num_boundary = hyperparams.get('num_boundary', 4000)
num_data = hyperparams.get('num_data', 10000)  # Number of data points within the domain
true_M = hyperparams['source_term_initial_M']

# Initial Condition Data (t = t_min)
X_ic = X_full[X_full[:, 2] == t_min]
v_ic = v_full[X_full[:, 2] == t_min]
# Randomly sample num_ic points
if len(X_ic) >= num_ic:
    ic_indices = np.random.choice(len(X_ic), num_ic, replace=False)
else:
    ic_indices = np.random.choice(len(X_ic), num_ic, replace=True)
X_ic_sampled = X_ic[ic_indices]
v_ic_sampled = v_ic[ic_indices]

# Data Points within the Domain
# Exclude initial time to avoid duplicating IC data
X_data_domain = X_full[X_full[:, 2] > t_min]
v_data_domain = v_full[X_full[:, 2] > t_min]
# Randomly sample num_data points
if len(X_data_domain) >= num_data:
    data_indices = np.random.choice(len(X_data_domain), num_data, replace=False)
else:
    data_indices = np.random.choice(len(X_data_domain), num_data, replace=True)
X_data_sampled = X_data_domain[data_indices]
v_data_sampled = v_data_domain[data_indices]

# Collocation Points (Randomly sample within the domain)
X_collocation = np.random.uniform([x_min, y_min, t_min], [x_max, y_max, t_max], (num_collocation, 3))

# Boundary Condition Data
def generate_boundary_points(x_min, x_max, y_min, y_max, t_min, t_max, num_points):
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
# Initialize the Inverse PINN Solver Using Hyperparameters
# =============================================================================

# Initialize the inverse PINN solver using hyperparameters
pinn = InverseMonodomainSolverPINNs(
    num_inputs=hyperparams['num_inputs'],  # For x, y, t
    num_layers=hyperparams['num_layers'],
    num_neurons=hyperparams['num_neurons'],
    device=hyperparams['device'],
    source_term_func=source_term_func_pinns,  # Use PINNs source term function
    initial_M=hyperparams['initial_M'],  # Starting guess for M
    loss_weights=hyperparams['loss_weights']
)

# =============================================================================
# Convert Data to Tensors and Assign to Model
# =============================================================================

# Convert initial condition data to tensors
pinn.X_ic = torch.tensor(X_ic_sampled, dtype=torch.float32).to(hyperparams['device'])
pinn.expected_u0 = torch.tensor(v_ic_sampled, dtype=torch.float32).unsqueeze(-1).to(hyperparams['device'])

# Convert collocation points to tensors
pinn.X_collocation = torch.tensor(X_collocation, dtype=torch.float32).to(hyperparams['device'])

# Convert boundary condition data to tensors
pinn.X_boundary = torch.tensor(X_boundary, dtype=torch.float32).to(hyperparams['device'])
pinn.normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32).to(hyperparams['device'])

# Convert domain data points to tensors
pinn.X_data = torch.tensor(X_data_sampled, dtype=torch.float32).to(hyperparams['device'])
pinn.expected_data = torch.tensor(v_data_sampled, dtype=torch.float32).unsqueeze(-1).to(hyperparams['device'])

# =============================================================================
# Print Model Parameters to Verify M is Included
# =============================================================================

print("Model parameters:")
for name, param in pinn.named_parameters():
    print(f"{name}: {param.shape}")

# Verify that M is included in the model parameters
print("\nVerifying that M is included in model parameters...")
M_in_parameters = any('M' in name for name, _ in pinn.named_parameters())
if M_in_parameters:
    print("M is included in the model parameters.")
else:
    print("Warning: M is not included in the model parameters.")

# =============================================================================
# Define the Optimizer Using Hyperparameters
# =============================================================================

# Define your optimizer using hyperparameters
#optimizer = torch.optim.Adam(pinn.parameters(), lr=hyperparams['learning_rate'])

optimizer = torch.optim.Adam(list(pinn.parameters()) + [pinn.M], lr=hyperparams['learning_rate'])

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

num_epochs = hyperparams['total_epochs']
batch_size = hyperparams['batch_size']

print("\nStarting training loop...")
start_time_training = time.time()

for epoch in range(num_epochs):
    # Assuming pinn.train_step is defined to return the respective losses and total loss
    try:
        pde_loss, IC_loss, BC_loss, data_loss, ode_loss, total_loss = pinn.train_step(optimizer, batch_size=batch_size)
    except Exception as e:
        print(f"Error during training at epoch {epoch}: {e}")
        break

    # Record losses and M estimate
    total_loss_list.append(total_loss)
    M_estimates.append(pinn.M.item())
    pde_loss_list.append(pde_loss)
    IC_loss_list.append(IC_loss)
    BC_loss_list.append(BC_loss)
    data_loss_list.append(data_loss)

    # Every 100 epochs, print the losses and current estimate of M
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Loss: {total_loss:.4e}, "
              f"PDE Loss: {pde_loss:.4e}, "
              f"IC Loss: {IC_loss:.4e}, "
              f"BC Loss: {BC_loss:.4e}, "
              f"Data Loss: {data_loss:.4e}, "
              f"M: {pinn.M.item():.4e}")

end_time_training = time.time()
print(f"\nTraining complete in {end_time_training - start_time_training:.2f} seconds\n")

# =============================================================================
# Saving Hyperparameters, Model, and Training Errors
# =============================================================================

# Save the training losses and M estimates to a file
training_results_filename = os.path.join(results_dir, 'training_results.npz')
np.savez_compressed(
    training_results_filename,
    total_loss_list=total_loss_list,
    pde_loss_list=pde_loss_list,
    IC_loss_list=IC_loss_list,
    BC_loss_list=BC_loss_list,
    data_loss_list=data_loss_list,
    M_estimates=M_estimates
)
print(f"Training results saved to {training_results_filename}")

# Save the hyperparameters to a JSON file
hyperparams_filename = os.path.join(results_dir, 'hyperparameters.json')
with open(hyperparams_filename, 'w') as f:
    json.dump(hyperparams, f, indent=4)
print(f"Hyperparameters saved to {hyperparams_filename}")

# Save the trained model parameters
model_filename = os.path.join(results_dir, 'trained_pinn_model.pth')
torch.save(pinn.state_dict(), model_filename)
print(f"Trained model saved to {model_filename}")

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

# Plot PDE and Data Loss Over Epochs
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

print("Training metrics plots saved.")
