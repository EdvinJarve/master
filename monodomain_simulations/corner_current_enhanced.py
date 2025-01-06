# =============================================================================
# Imports
# =============================================================================

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

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import qmc
import torch.optim as optim

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import EnhancedMonodomainSolverPINNs

# Define the results directory relative to the project root
results_dir = os.path.join(project_root, 'monodomain_results', 'corner_case_enhanced')
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# Function Definitions
# =============================================================================

# Define source term function for PINNs
def source_term_func_pinns(x_spatial, t):
    """
    Gaussian stimulus applied in the upper left corner between x ∈ [0,0.2] and y ∈ [0.8,1],
    smoothed with a Gaussian distribution. The stimulus is turned on at t=0.05 and off at t=0.2.

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
    t_on = 0.05
    t_off = 0.2

    # Gaussian distribution
    gaussian = 50 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Time window: 1 when t_on <= t <= t_off, else 0
    time_window = ((t >= t_on) & (t <= t_off)).float()

    return gaussian * time_window

# Define source term function for FEM using UFL
def source_term_func(x, y, t):
    """
    Gaussian stimulus applied in the upper left corner between x ∈ [0,0.2] and y ∈ [0.8,1],
    smoothed with a Gaussian distribution. The stimulus is turned on at t=0.05 and off at t=0.2.

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
    t_on = 0.05
    t_off = 0.2

    # Gaussian distribution
    gaussian = 50 * ufl.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Time window: 1 when t_on <= t <= t_off, else 0
    time_window = ufl.conditional(
        ufl.And(t >= t_on, t <= t_off),
        1.0,
        0.0
    )

    return gaussian * time_window

# =============================================================================
# FEM Simulation
# =============================================================================

# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 100, 100, 100  # Spatial and temporal resolution
T = 1.0                  # Final time
dt = T / Nt              # Time step size
M = 1.0                  # Conductivity

# Create mesh
domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)

# Print statements for simulation steps
print("Solving simulation with FEM")
start_time_fem = time.time()

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
time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

# Run FEM simulation with time_points to obtain solutions_fem
errors_fem, computation_time_fem, solutions_fem = sim_fem.run(time_points=time_points)
end_time_fem = time.time()
print(f"FEM simulation complete in {end_time_fem - start_time_fem:.2f} seconds\n")

# Extract coordinates of degrees of freedom
dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]


# =============================================================================
# PINNs Simulation
# =============================================================================

print("Solving simulation with PINNs")

# Specify device
device = 'cpu'
print(f"Using device: {device}\n")

# Initialize the PINNs model with increased capacity and adjusted loss weights
model = EnhancedMonodomainSolverPINNs(
    num_inputs=3,          # For example, spatial coordinate x and time t
    num_layers=2,          # Number of hidden layers
    num_neurons=512,        # Number of neurons per hidden layer
    device='cpu',         
    source_term_func=source_term_func_pinns,
    M=1.0,                 # Parameter M in the monodomain equation
    use_ode=False,         # Whether to use ODE for current terms
    ode_func=None,         # ODE function if use_ode is True
    n_state_vars=0,        # Number of state variables if using ODE
    loss_function='L2',    # Type of loss function
    use_fourier=True,      # Enable Fourier Feature Embeddings
    fourier_dim=64,       # Number of Fourier features (m)
    sigma=1.0,             # Scale parameter for Fourier embeddings (within [1, 10])
    use_rwf=True,          # Enable Random Weight Factorization
    mu=1.0,                # Mean for RWF scale factors
    sigma_rwf=0.1          # Std dev for RWF scale factors
)

# =============================================================================
# Data Generation Using Latin Hypercube Sampling
# =============================================================================

# Define domain boundaries
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, T  # T is 1.0

# Define the number of points
N_collocation = 20000
N_ic = 4000
N_bc = 4000
N_val = 2000
N_test = 2000

# Define the number of validation points for collocation, IC, and BC
N_collocation_val = 1000
N_ic_val = 100
N_bc_val = 100

# =============================================================================
# Helper Function for Sampling within Bounds
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

# =============================================================================
# Generate Collocation Points with 50% in Stimulus Region
# =============================================================================

# Define stimulus region bounds
stimulus_bounds = [(0.0, 0.2),    # x ∈ [0, 0.2]
                  (0.8, 1.0),    # y ∈ [0.8, 1]
                  (0.05, 0.2)]   # t ∈ [0.05, 0.2]

# Define rest region bounds (entire domain)
rest_bounds = [(0.0, 1.0),      # x ∈ [0, 1]
              (0.0, 1.0),      # y ∈ [0, 1]
              (0.0, 1.0)]      # t ∈ [0, 1]

# Number of collocation points in each region
N_collocation_stimulus = N_collocation // 2  # 50%
N_collocation_rest = N_collocation - N_collocation_stimulus  # Remaining 50%

# Sample collocation points in stimulus region
X_collocation_stimulus = sample_lhs(N_collocation_stimulus, stimulus_bounds)

# Sample collocation points in rest region
X_collocation_rest = sample_lhs(N_collocation_rest, rest_bounds)

# Combine collocation points
X_collocation = np.vstack([X_collocation_stimulus, X_collocation_rest])

# Shuffle the combined collocation points to ensure randomness
np.random.shuffle(X_collocation)

# =============================================================================
# Generate Validation Collocation Points with 50% in Stimulus Region
# =============================================================================

# Number of validation collocation points in each region
N_collocation_val_stimulus = N_collocation_val // 2  # 50%
N_collocation_val_rest = N_collocation_val - N_collocation_val_stimulus  # Remaining 50%

# Sample validation collocation points in stimulus region
X_collocation_val_stimulus = sample_lhs(N_collocation_val_stimulus, stimulus_bounds)

# Sample validation collocation points in rest region
X_collocation_val_rest = sample_lhs(N_collocation_val_rest, rest_bounds)

# Combine validation collocation points
X_collocation_val = np.vstack([X_collocation_val_stimulus, X_collocation_val_rest])

# Shuffle the combined validation collocation points
np.random.shuffle(X_collocation_val)

# =============================================================================
# Generate Initial Condition Points
# =============================================================================
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)

X_ic = sample_ic.copy()
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]  # x
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]  # y

# Add t=0
X_ic = np.hstack((X_ic, np.zeros((N_ic, 1))))

# Set expected initial condition values to zero
expected_u0 = np.zeros((N_ic, 1))

# Generate validation IC points
sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)

X_ic_val = sample_ic_val.copy()
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]  # x
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]  # y

# Add t=0
X_ic_val = np.hstack((X_ic_val, np.zeros((N_ic_val, 1))))

# Set expected initial condition values for validation to zero
expected_u0_val = np.zeros((N_ic_val, 1))

# =============================================================================
# Generate Boundary Condition Points
# =============================================================================
N_per_boundary = N_bc // 4

# Training Boundary Points
def generate_boundary_points(N_per_boundary, x_fixed=None, y_fixed=None):
    sampler_boundary = qmc.LatinHypercube(d=2)
    sample_boundary = sampler_boundary.random(n=N_per_boundary)
    X_boundary = np.zeros((N_per_boundary, 3))
    if x_fixed is not None:
        X_boundary[:, 0] = x_fixed
        X_boundary[:, 1] = y_min + (y_max - y_min) * sample_boundary[:, 0]  # y
    elif y_fixed is not None:
        X_boundary[:, 0] = x_min + (x_max - x_min) * sample_boundary[:, 0]  # x
        X_boundary[:, 1] = y_fixed
    X_boundary[:, 2] = t_min + (t_max - t_min) * sample_boundary[:, 1]  # t
    return X_boundary

X_left = generate_boundary_points(N_per_boundary, x_fixed=x_min)
X_right = generate_boundary_points(N_per_boundary, x_fixed=x_max)
X_bottom = generate_boundary_points(N_per_boundary, y_fixed=y_min)
X_top = generate_boundary_points(N_per_boundary, y_fixed=y_max)

# Combine all boundary points
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

# Define normal vectors
normal_vectors_left = np.tile(np.array([[-1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_right = np.tile(np.array([[1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_bottom = np.tile(np.array([[0.0, -1.0]]), (N_per_boundary, 1))
normal_vectors_top = np.tile(np.array([[0.0, 1.0]]), (N_per_boundary, 1))

# Combine all normal vectors
normal_vectors = np.vstack([normal_vectors_left, normal_vectors_right, normal_vectors_bottom, normal_vectors_top])

# Validation Boundary Points
N_per_boundary_val = N_bc_val // 4

X_left_val = generate_boundary_points(N_per_boundary_val, x_fixed=x_min)
X_right_val = generate_boundary_points(N_per_boundary_val, x_fixed=x_max)
X_bottom_val = generate_boundary_points(N_per_boundary_val, y_fixed=y_min)
X_top_val = generate_boundary_points(N_per_boundary_val, y_fixed=y_max)

# Combine all validation boundary points
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

# Define normal vectors for validation
normal_vectors_left_val = np.tile(np.array([[-1.0, 0.0]]), (N_per_boundary_val, 1))
normal_vectors_right_val = np.tile(np.array([[1.0, 0.0]]), (N_per_boundary_val, 1))
normal_vectors_bottom_val = np.tile(np.array([[0.0, -1.0]]), (N_per_boundary_val, 1))
normal_vectors_top_val = np.tile(np.array([[0.0, 1.0]]), (N_per_boundary_val, 1))

# Combine all validation normal vectors
normal_vectors_val = np.vstack([normal_vectors_left_val, normal_vectors_right_val, normal_vectors_bottom_val, normal_vectors_top_val])

# =============================================================================
# Generate Test Data
# =============================================================================
# Since there's no analytical solution, test data should be aligned with FEM simulation points

# Generate test points aligned with FEM mesh and time points
# Typically, you would use the FEM mesh points at different time points
# For simplicity, we can reuse the FEM solutions' spatial points at different times

# =============================================================================
# Normalize Data
# =============================================================================

def normalize_data(X, x_min, x_max, y_min, y_max, t_min, t_max):
    """
    Normalize spatial and temporal coordinates to [0, 1].

    Args:
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Normalized data.
    """
    X_norm = X.copy()
    X_norm[:, 0] = (X[:, 0] - x_min) / (x_max - x_min)  # x
    X_norm[:, 1] = (X[:, 1] - y_min) / (y_max - y_min)  # y
    X_norm[:, 2] = (X[:, 2] - t_min) / (t_max - t_min)  # t
    return X_norm

# Apply normalization to all datasets
X_collocation_norm = normalize_data(X_collocation, x_min, x_max, y_min, y_max, t_min, t_max)
X_collocation_val_norm = normalize_data(X_collocation_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_ic_norm = normalize_data(X_ic, x_min, x_max, y_min, y_max, t_min, t_max)
X_ic_val_norm = normalize_data(X_ic_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_norm = normalize_data(X_boundary, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_val_norm = normalize_data(X_boundary_val, x_min, x_max, y_min, y_max, t_min, t_max)
# X_test_norm is not used since we compare against FEM solutions

# =============================================================================
# Convert Data to PyTorch Tensors and Move to Device
# =============================================================================

# Convert to tensors
X_collocation_tensor = torch.tensor(X_collocation_norm, dtype=torch.float32).to(device)
X_ic_tensor = torch.tensor(X_ic_norm, dtype=torch.float32).to(device)
expected_u0_tensor = torch.tensor(expected_u0, dtype=torch.float32).to(device)
X_boundary_tensor = torch.tensor(X_boundary_norm, dtype=torch.float32).to(device)
normal_vectors_tensor = torch.tensor(normal_vectors, dtype=torch.float32).to(device)

X_collocation_val_tensor = torch.tensor(X_collocation_val_norm, dtype=torch.float32).to(device)
X_ic_val_tensor = torch.tensor(X_ic_val_norm, dtype=torch.float32).to(device)
expected_u0_val_tensor = torch.tensor(expected_u0_val, dtype=torch.float32).to(device)
X_boundary_val_tensor = torch.tensor(X_boundary_val_norm, dtype=torch.float32).to(device)
normal_vectors_val_tensor = torch.tensor(normal_vectors_val, dtype=torch.float32).to(device)

# Assign data to the model
model.X_collocation = X_collocation_tensor
model.X_ic = X_ic_tensor
model.expected_u0 = expected_u0_tensor
model.X_boundary = X_boundary_tensor
model.normal_vectors = normal_vectors_tensor
model.X_data = None  # Assuming no additional data; set if available
model.expected_data = None  # Set if X_data is provided

# =============================================================================
# Define Optimizer, Scheduler, and Training Parameters
# =============================================================================

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define learning rate scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=500,
                                                 verbose=True)

# Define training parameters
epochs = 30000
batch_size = 256  # Set to None to use entire dataset

# Initialize variables for tracking progress
loss_list = []
val_loss_list = []
epoch_list = []
best_val_loss = float('inf')
no_improve_counter = 0
patience = 1000  # For early stopping
best_model_path = os.path.join(results_dir, 'best_model.pth')

# =============================================================================
# Training Loop with Validation
# =============================================================================

print("Starting PINNs training...")
start_time_pinns = time.time()
for epoch in range(epochs + 1):
    # Perform a training step
    pde_loss, IC_loss, BC_loss, data_loss, ode_loss, total_loss = model.train_step(optimizer, batch_size)
    
    # Validation
    model.eval()
    # Ensure gradients are enabled by not using torch.no_grad()
    total_val_loss = model.validate(
        X_collocation_val=X_collocation_val_tensor,
        X_ic_val=X_ic_val_tensor,
        expected_u0_val=expected_u0_val_tensor,
        X_boundary_val=X_boundary_val_tensor,
        normal_vectors_val=normal_vectors_val_tensor
    )
    
    # Update learning rate scheduler based on validation loss
    scheduler.step(total_val_loss)
    
    if epoch % 100 == 0:
        # Append training loss for plotting
        loss_list.append(total_loss)
        epoch_list.append(epoch)
    
        # Append validation loss for plotting
        val_loss_list.append(total_val_loss)
    
        # Print training and validation losses
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, '
              f'BC Loss: {BC_loss:.4e}, ODE Loss: {ode_loss:.4e}, '
              f'Total Loss: {total_loss:.4e}, Validation Loss: {total_val_loss:.4e}\n')
    
        # Check if validation loss improved
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            no_improve_counter = 0
            # Save the model
            model.save_model(best_model_path)
            print(f"New best model saved with validation loss {best_val_loss:.4e}")
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print("Early stopping triggered.")
                break

end_time_pinns = time.time()
computation_time_pinns = end_time_pinns - start_time_pinns
print(f"PINNs training complete in {computation_time_pinns:.2f} seconds\n")

# =============================================================================
# Load the Best Model and Evaluate on Test Data
# =============================================================================

# Load the best model after training
model.load_model(best_model_path)

# Since there's no analytical solution, evaluate PINNs against FEM at mesh points and times
# We'll use the FEM solutions for comparison

# Define test times (same as simulation time points)
test_times = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

for t in test_times:
    if t not in solutions_fem:
        print(f"FEM solution for time t = {t} not found. Skipping.")
        continue
    
    # Extract FEM solution at time t
    fem_v = solutions_fem[t]  # Shape: (N,)
    
    # Create input tensor for PINNs evaluation on the FEM mesh
    X_test_t = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))  # Shape: (N, 3)
    X_test_t_norm = normalize_data(X_test_t, x_min, x_max, y_min, y_max, t_min, t_max)  # Normalized
    X_test_t_tensor = torch.tensor(X_test_t_norm, dtype=torch.float32).to(device)  # Shape: (N, 3)
    
    # Evaluate PINNs model
    with torch.no_grad():
        y_pred_test = model.evaluate(X_test_t_tensor)  # Shape: (N, 1)
        y_pred_test_np = y_pred_test.cpu().numpy().reshape(-1)  # Shape: (N,)
    
    # Compute error metrics between PINNs and FEM
    mse_test = mean_squared_error(fem_v, y_pred_test_np)
    mae_test = mean_absolute_error(fem_v, y_pred_test_np)
    rmse_test = np.sqrt(mse_test)
    
    print(f"Time {t}: Test MSE: {mse_test:.4e}, MAE: {mae_test:.4e}, RMSE: {rmse_test:.4e}")

# =============================================================================
# Plot Training and Validation Loss Over Epochs
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(epoch_list, loss_list, label='Training Loss')
plt.plot(epoch_list, val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_plot.png'), dpi=300)
plt.close()
print("Loss plot saved.")

# =============================================================================
# Plot Comparisons Between PINNs and FEM
# =============================================================================

def plot_comparisons(model, solutions_fem, comparison_times, results_dir, device='cpu'):
    """
    Plot PINNs predictions, FEM solutions, and their absolute errors in a 2x2 subplot layout.

    Parameters:
        model (MonodomainSolverPINNs): Trained PINNs model.
        solutions_fem (dict): Dictionary with time keys mapping to FEM solutions.
        comparison_times (list): List of time points to compare.
        results_dir (str): Directory to save plots.
        device (str): Computational device ('cpu' or 'cuda').
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import os

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Access global variables
    global x_coords, y_coords, x_min, x_max, y_min, y_max, t_min, t_max, normalize_data

    # Create triangulation for tricontourf
    triang = Triangulation(x_coords, y_coords)

    for t in comparison_times:
        print(f"Comparing at time t = {t}")

        # Check if FEM solution exists for time t
        if t not in solutions_fem:
            print(f"FEM solution for time t = {t} not found. Skipping.")
            continue

        # Extract FEM solution at time t
        fem_v = solutions_fem[t]  # Shape: (N,)

        # Create input tensor for PINNs evaluation on the FEM mesh
        X_pinns_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))  # Shape: (N, 3)
        X_pinns_eval_norm = normalize_data(X_pinns_eval, x_min, x_max, y_min, y_max, t_min, t_max)  # Normalized
        X_pinns_eval_tensor = torch.tensor(X_pinns_eval_norm, dtype=torch.float32).to(device)  # Shape: (N, 3)

        # Evaluate PINNs model
        with torch.no_grad():
            y_pinns_pred = model.evaluate(X_pinns_eval_tensor)  # Shape: (N, 1)
            y_pinns_pred_np = y_pinns_pred.cpu().numpy().reshape(-1)  # Shape: (N,)

        # Compute absolute error between PINNs and FEM
        error = np.abs(y_pinns_pred_np - fem_v)

        # Compute error metrics
        mse = mean_squared_error(fem_v, y_pinns_pred_np)
        mae = mean_absolute_error(fem_v, y_pinns_pred_np)
        rmse = np.sqrt(mse)

        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparisons at t = {t}', fontsize=16)

        # Subplot 1: FEM Prediction
        cs1 = axs[0, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
        cbar1 = fig.colorbar(cs1, ax=axs[0, 0])
        cbar1.set_label('FEM Prediction')
        axs[0, 0].set_title('FEM Prediction')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')

        # Subplot 2: PINNs Prediction
        cs2 = axs[0, 1].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
        cbar2 = fig.colorbar(cs2, ax=axs[0, 1])
        cbar2.set_label('PINNs Prediction')
        axs[0, 1].set_title('PINNs Prediction')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')

        # Subplot 3: Absolute Error (|PINNs - FEM|)
        cs3 = axs[1, 0].tricontourf(triang, error, levels=50, cmap='viridis')
        cbar3 = fig.colorbar(cs3, ax=axs[1, 0])
        cbar3.set_label('Absolute Error |PINNs - FEM|')
        axs[1, 0].set_title('Absolute Error')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')

        # Subplot 4: Source Term at time t (optional)
        # Since the source term is zero outside the stimulus region, it's useful to visualize it
        # Create input tensor for source term evaluation
        x_vals = x_coords
        y_vals = y_coords
        x_spatial = np.column_stack((x_vals, y_vals))
        t_vals = np.full_like(x_vals, t)
        t_tensor = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32, device=device)
        x_spatial_tensor = torch.tensor(x_spatial, dtype=torch.float32, device=device)
        source_term_vals = source_term_func_pinns(x_spatial_tensor, t_tensor).cpu().numpy().reshape(-1)

        cs4 = axs[1, 1].tricontourf(triang, source_term_vals, levels=50, cmap='viridis')
        cbar4 = fig.colorbar(cs4, ax=axs[1, 1])
        cbar4.set_label('Source Term')
        axs[1, 1].set_title('Source Term')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the suptitle

        # Save the figure
        plot_filename = f'comparison_t_{t}.png'
        plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
        plt.close()
        print(f"Plots saved for time t = {t}\n")

        # Print error metrics
        print(f"Time {t}:")
        print(f"  PINNs - MSE: {mse:.4e}, MAE: {mae:.4e}, RMSE: {rmse:.4e}")
        print(f"  Plots saved as {plot_filename}\n")

# =============================================================================
# Execute Plotting Function
# =============================================================================

# Define comparison time points
comparison_times = [0.0, 0.1, 0.15, 0.2, 0.4, 0.8, 1.0]

# Call the plotting function
plot_comparisons(model, solutions_fem, comparison_times, results_dir, device=device)

print("Comparison between FEM and PINNs simulations complete.")

"""
Time 0.0: Test MSE: 1.8782e-04, MAE: 1.0102e-02, RMSE: 1.3705e-02
Time 0.1: Test MSE: 8.2765e-05, MAE: 6.3069e-03, RMSE: 9.0975e-03
Time 0.2: Test MSE: 8.2636e-04, MAE: 1.7769e-02, RMSE: 2.8746e-02
Time 0.4: Test MSE: 1.6983e-05, MAE: 3.7190e-03, RMSE: 4.1210e-03
Time 0.8: Test MSE: 1.9633e-05, MAE: 3.9622e-03, RMSE: 4.4309e-03
Time 1.0: Test MSE: 2.4770e-05, MAE: 4.6500e-03, RMSE: 4.9770e-03
Loss plot saved.
Comparing at time t = 0.0
Plots saved for time t = 0.0

Time 0.0:
  PINNs - MSE: 1.8782e-04, MAE: 1.0102e-02, RMSE: 1.3705e-02
  Plots saved as comparison_t_0.0.png

Comparing at time t = 0.1
Plots saved for time t = 0.1

Time 0.1:
  PINNs - MSE: 8.2765e-05, MAE: 6.3069e-03, RMSE: 9.0975e-03
  Plots saved as comparison_t_0.1.png

Comparing at time t = 0.15
FEM solution for time t = 0.15 not found. Skipping.
Comparing at time t = 0.2
Plots saved for time t = 0.2

Time 0.2:
  PINNs - MSE: 8.2636e-04, MAE: 1.7769e-02, RMSE: 2.8746e-02
  Plots saved as comparison_t_0.2.png

Comparing at time t = 0.4
Plots saved for time t = 0.4

Time 0.4:
  PINNs - MSE: 1.6983e-05, MAE: 3.7190e-03, RMSE: 4.1210e-03
  Plots saved as comparison_t_0.4.png

Comparing at time t = 0.8
Plots saved for time t = 0.8

Time 0.8:
  PINNs - MSE: 1.9633e-05, MAE: 3.9622e-03, RMSE: 4.4309e-03
  Plots saved as comparison_t_0.8.png

Comparing at time t = 1.0
Plots saved for time t = 1.0

Time 1.0:
  PINNs - MSE: 2.4770e-05, MAE: 4.6500e-03, RMSE: 4.9770e-03
  Plots saved as comparison_t_1.0.png

Comparison between FEM and PINNs simulations complete.


"""