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
from scipy.stats import qmc

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
results_dir = os.path.join(project_root, 'monodomain_results', 'non_iso_corner')
os.makedirs(results_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Hyperparameters Definition
# =============================================================================

hyperparams = {
    'num_inputs': 3,             # Number of inputs to the PINN (x, y, t)
    'num_layers': 2,             # Number of hidden layers in the neural network
    'num_neurons': 256,          # Number of neurons per hidden layer
    'device': 'cpu',  # Device configuration
    'initial_M': [1e-2, 1e-2],     # Initial guesses for M_xx and M_yy
    'learning_rate': 1e-4,       # Learning rate for the optimizer
    'pde_epochs': 0,          # Number of epochs for PDE-only training
    'total_epochs': 60000,       # Total number of training epochs
    'batch_size': 64,           # Batch size for training
    'loss_weights': {            # Weights for different loss components
        'pde_loss': 1.0,
        'IC_loss': 0.333,
        'BC_loss': 0.333,
        'data_loss': 1000.0,     # Increased weight for data loss
        'ode_loss': 0.0          # Set to 0 if not using ODE
    },
    'source_term_initial_M': [2.0, 1.0],  # True values of M_xx and M_yy used in FEM simulation
    'num_ic': 4000,
    'num_collocation': 10000,
    'num_boundary': 4000,
    'num_data': 5000             # Number of data points within the domain
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
    M_xx_true, M_yy_true = hyperparams['source_term_initial_M']  # True conductivity tensor components

    # Create mesh
    domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)

    # Define conductivity tensor M_i as a UFL expression
    M_i = ufl.as_tensor([[M_xx_true, 0.0],
                        [0.0, M_yy_true]])

    # Initialize and run the FEM simulation with the MonodomainSolverFEM class
    sim_fem = MonodomainSolverFEM(
        mesh=domain_mesh,
        T=T,
        dt=dt,
        M_i=M_i,  # Pass the tensor directly
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
    # Plotting the FEM simulation results at specific time steps
    # =============================================================================

    print("Plotting FEM simulation results...")
    time_steps_to_plot = [0, int(Nt/4), int(Nt/2), int(3*Nt/4), Nt]
    for i in time_steps_to_plot:
        plt.figure(figsize=(8, 6))
        plt.tricontourf(x_coords, y_coords, solutions_fem_array[i], levels=50)
        plt.colorbar()
        plt.title(f'FEM Solution at t = {time_points[i]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(os.path.join(results_dir, f'fem_solution_t_{time_points[i]:.2f}.png'))
        plt.close()
    print("FEM simulation plots saved.\n")

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

# Data Points within the Domain
# Exclude initial time to avoid duplicating IC data
X_data_domain = X_full[X_full[:, 2] > 0.0]
v_data_domain = v_full[X_full[:, 2] > 0.0]

# Randomly sample num_data points
num_data = hyperparams.get('num_data', 5000)
if len(X_data_domain) >= num_data:
    data_indices = np.random.choice(len(X_data_domain), num_data, replace=False)
else:
    data_indices = np.random.choice(len(X_data_domain), num_data, replace=True)
X_data_sampled = X_data_domain[data_indices]
v_data_sampled = v_data_domain[data_indices]

# Convert data points to tensors
X_data = torch.tensor(X_data_sampled, dtype=torch.float32)
expected_data = torch.tensor(v_data_sampled, dtype=torch.float32).unsqueeze(-1)

# =============================================================================
# Data Generation for PINN Training
# =============================================================================

def sample_lhs(n_samples, bounds):
    """
    Sample n_samples using Latin Hypercube Sampling within the specified bounds.

    Parameters:
        n_samples (int): Number of samples to generate.
        bounds (list of tuples): Bounds for each dimension, e.g., [(x_min, x_max), (y_min, y_max), (t_min, t_max)].

    Returns:
        np.ndarray: Array of shape (n_samples, len(bounds)) with sampled points.
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n=n_samples)
    # Scale samples to the bounds
    X = np.empty_like(sample)
    for i, (lower, upper) in enumerate(bounds):
        X[:, i] = lower + (upper - lower) * sample[:, i]
    return X

def generate_boundary_points(n_samples):
    """Generate boundary points using LHS sampling."""
    # Sample points for each boundary
    n_per_boundary = n_samples // 4
    
    # Left boundary (x = 0)
    left_sampler = qmc.LatinHypercube(d=2)
    left_samples = left_sampler.random(n=n_per_boundary)
    left_boundary = np.zeros((n_per_boundary, 3))
    left_boundary[:, 0] = 0.0  # x = 0
    left_boundary[:, 1] = y_min + (y_max - y_min) * left_samples[:, 0]  # y
    left_boundary[:, 2] = t_min + (t_max - t_min) * left_samples[:, 1]  # t
    
    # Right boundary (x = 1)
    right_sampler = qmc.LatinHypercube(d=2)
    right_samples = right_sampler.random(n=n_per_boundary)
    right_boundary = np.zeros((n_per_boundary, 3))
    right_boundary[:, 0] = 1.0  # x = 1
    right_boundary[:, 1] = y_min + (y_max - y_min) * right_samples[:, 0]  # y
    right_boundary[:, 2] = t_min + (t_max - t_min) * right_samples[:, 1]  # t
    
    # Bottom boundary (y = 0)
    bottom_sampler = qmc.LatinHypercube(d=2)
    bottom_samples = bottom_sampler.random(n=n_per_boundary)
    bottom_boundary = np.zeros((n_per_boundary, 3))
    bottom_boundary[:, 0] = x_min + (x_max - x_min) * bottom_samples[:, 0]  # x
    bottom_boundary[:, 1] = 0.0  # y = 0
    bottom_boundary[:, 2] = t_min + (t_max - t_min) * bottom_samples[:, 1]  # t
    
    # Top boundary (y = 1)
    top_sampler = qmc.LatinHypercube(d=2)
    top_samples = top_sampler.random(n=n_per_boundary)
    top_boundary = np.zeros((n_per_boundary, 3))
    top_boundary[:, 0] = x_min + (x_max - x_min) * top_samples[:, 0]  # x
    top_boundary[:, 1] = 1.0  # y = 1
    top_boundary[:, 2] = t_min + (t_max - t_min) * top_samples[:, 1]  # t
    
    # Combine all boundaries
    X_boundary = np.vstack([left_boundary, right_boundary, bottom_boundary, top_boundary])
    
    # Compute normal vectors
    normals = np.zeros((4 * n_per_boundary, 2))
    normals[0:n_per_boundary] = [-1, 0]  # Left boundary
    normals[n_per_boundary:2*n_per_boundary] = [1, 0]  # Right boundary
    normals[2*n_per_boundary:3*n_per_boundary] = [0, -1]  # Bottom boundary
    normals[3*n_per_boundary:4*n_per_boundary] = [0, 1]  # Top boundary
    
    return X_boundary, normals

# Define domain bounds
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Define stimulus region bounds
stimulus_bounds = [(0.0, 0.2),    # x ∈ [0, 0.2]
                  (0.8, 1.0),    # y ∈ [0.8, 1]
                  (0.05, 0.2)]   # t ∈ [0.05, 0.2]

# Define rest region bounds (entire domain)
rest_bounds = [(x_min, x_max),      # x ∈ [0, 1]
              (y_min, y_max),      # y ∈ [0, 1]
              (t_min, t_max)]      # t ∈ [0, 1]

# Number of points for each region
N_collocation = hyperparams['num_collocation']
N_collocation_stimulus = N_collocation // 2  # 50% in stimulus region
N_collocation_rest = N_collocation - N_collocation_stimulus

# Sample collocation points
X_collocation_stimulus = sample_lhs(N_collocation_stimulus, stimulus_bounds)
X_collocation_rest = sample_lhs(N_collocation_rest, rest_bounds)
X_collocation = np.vstack([X_collocation_stimulus, X_collocation_rest])
np.random.shuffle(X_collocation)

# Sample validation collocation points
N_collocation_val = N_collocation // 5  # 20% of training points for validation
N_collocation_val_stimulus = N_collocation_val // 2
N_collocation_val_rest = N_collocation_val - N_collocation_val_stimulus

X_collocation_val_stimulus = sample_lhs(N_collocation_val_stimulus, stimulus_bounds)
X_collocation_val_rest = sample_lhs(N_collocation_val_rest, rest_bounds)
X_collocation_val = np.vstack([X_collocation_val_stimulus, X_collocation_val_rest])
np.random.shuffle(X_collocation_val)

# Generate Initial Condition Points
N_ic = hyperparams['num_ic']
N_ic_val = N_ic // 5  # 20% of training points for validation

# Sample IC points using LHS
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)
X_ic = np.zeros((N_ic, 3))
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]  # x
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]  # y
# t=0 is already set by np.zeros

# Set expected initial condition values
expected_u0 = np.zeros((N_ic, 1))

# Generate validation IC points
sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)
X_ic_val = np.zeros((N_ic_val, 3))
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]  # x
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]  # y
# t=0 is already set by np.zeros

# Set expected initial condition values for validation
expected_u0_val = np.zeros((N_ic_val, 1))

# Generate boundary points using LHS
N_boundary = hyperparams.get('num_boundary', 4000)
X_boundary, normal_vectors = generate_boundary_points(N_boundary)

# Convert numpy arrays to torch tensors
X_collocation = torch.tensor(X_collocation, dtype=torch.float32)
X_collocation_val = torch.tensor(X_collocation_val, dtype=torch.float32)
X_ic = torch.tensor(X_ic, dtype=torch.float32)
expected_u0 = torch.tensor(expected_u0, dtype=torch.float32)
X_ic_val = torch.tensor(X_ic_val, dtype=torch.float32)
expected_u0_val = torch.tensor(expected_u0_val, dtype=torch.float32)
X_boundary = torch.tensor(X_boundary, dtype=torch.float32)
normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32)

# Convert domain data points to tensors
X_data = torch.tensor(X_data_sampled, dtype=torch.float32)
expected_data = torch.tensor(v_data_sampled, dtype=torch.float32).unsqueeze(-1)

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
    initial_M=hyperparams['initial_M'],  # Starting guess for M as a list [M_xx, M_yy]
    loss_weights=hyperparams['loss_weights']
)

# =============================================================================
# Convert Data to Tensors and Assign to Model
# =============================================================================

# Convert initial condition data to tensors
pinn.X_ic = X_ic.to(hyperparams['device'])
pinn.expected_u0 = expected_u0.to(hyperparams['device'])

# Convert collocation points to tensors
pinn.X_collocation = X_collocation.to(hyperparams['device'])

# Convert boundary condition data to tensors
pinn.X_boundary = X_boundary.to(hyperparams['device'])
pinn.normal_vectors = normal_vectors.to(hyperparams['device'])

# Convert domain data points to tensors
pinn.X_data = X_data.to(hyperparams['device'])
pinn.expected_data = expected_data.to(hyperparams['device'])

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
optimizer = torch.optim.Adam(list(pinn.parameters()) + [pinn.M], lr=hyperparams['learning_rate'])

# =============================================================================
# Training Loop
# =============================================================================

# Initialize lists to store losses and M estimates
total_loss_list = []
pde_loss_list = []
ic_loss_list = []
bc_loss_list = []
data_loss_list = []
M_xx_list = []
M_yy_list = []

print("\nStarting training...")
start_time = time.time()

# Phase 1: Train with PDE loss only
print("\nPhase 1: Training with PDE loss only...")
pde_only_weights = hyperparams['loss_weights'].copy()
pde_only_weights['data_loss'] = 0.0  # Set data loss weight to 0

for epoch in range(hyperparams['pde_epochs']):
    # Temporarily set the loss weights to PDE-only
    original_weights = pinn.loss_weights
    pinn.loss_weights = pde_only_weights
    
    # Perform training step
    losses = pinn.train_step(optimizer, hyperparams['batch_size'])
    pde_loss, ic_loss, bc_loss, data_loss, ode_loss, total_loss = losses
    
    # Restore original weights
    pinn.loss_weights = original_weights
    
    # Store losses and M estimates
    total_loss_list.append(total_loss)
    pde_loss_list.append(pde_loss)
    ic_loss_list.append(ic_loss)
    bc_loss_list.append(bc_loss)
    data_loss_list.append(data_loss)
    M_xx_list.append(pinn.M[0].item())
    M_yy_list.append(pinn.M[1].item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{hyperparams['pde_epochs']}], "
              f"Total Loss: {total_loss:.6f}, "
              f"PDE Loss: {pde_loss:.6f}, "
              f"IC Loss: {ic_loss:.6f}, "
              f"BC Loss: {bc_loss:.6f}, "
              f"M_xx: {pinn.M[0].item():.6f}, "
              f"M_yy: {pinn.M[1].item():.6f}")

# Phase 2: Train with both PDE and data loss
print("\nPhase 2: Training with both PDE and data loss...")
remaining_epochs = hyperparams['total_epochs'] - hyperparams['pde_epochs']

for epoch in range(remaining_epochs):
    # Perform training step with all losses
    losses = pinn.train_step(optimizer, hyperparams['batch_size'])
    pde_loss, ic_loss, bc_loss, data_loss, ode_loss, total_loss = losses
    
    # Store losses and M estimates
    total_loss_list.append(total_loss)
    pde_loss_list.append(pde_loss)
    ic_loss_list.append(ic_loss)
    bc_loss_list.append(bc_loss)
    data_loss_list.append(data_loss)
    M_xx_list.append(pinn.M[0].item())
    M_yy_list.append(pinn.M[1].item())
    

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{remaining_epochs}], "
              f"Total Loss: {total_loss:.4e}, "
              f"PDE Loss: {pde_loss:.4e}, "
              f"IC Loss: {ic_loss:.4e}, "
              f"BC Loss: {bc_loss:.4e}, "
              f"Data Loss: {data_loss:.4e}, "
              f"M_xx: {pinn.M[0].item():.4e}, "
              f"M_yy: {pinn.M[1].item():.4e}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# =============================================================================
# Saving Hyperparameters, Model, and Training Errors
# =============================================================================

# Save the training losses and M estimates to a file
training_results_filename = os.path.join(results_dir, 'training_results.npz')
np.savez_compressed(
    training_results_filename,
    total_loss_list=total_loss_list,
    pde_loss_list=pde_loss_list,
    ic_loss_list=ic_loss_list,
    bc_loss_list=bc_loss_list,
    data_loss_list=data_loss_list,
    M_xx_list=M_xx_list,
    M_yy_list=M_yy_list
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
M_estimates_array = np.array([M_xx_list, M_yy_list]).T  # Shape (num_epochs, 2)
true_M = hyperparams['source_term_initial_M']

plt.figure(figsize=(10, 5))
plt.plot(M_estimates_array[:, 0], label='Estimated M_xx', color='blue')
plt.plot(M_estimates_array[:, 1], label='Estimated M_yy', color='green')
plt.axhline(y=true_M[0], color='blue', linestyle='--', label='True M_xx')
plt.axhline(y=true_M[1], color='green', linestyle='--', label='True M_yy')
plt.xlabel('Epochs')
plt.ylabel('M Values')
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
