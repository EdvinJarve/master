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

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import EnhancedMonodomainSolverPINNs

# Define the results directory relative to the project root
results_dir = os.path.join(project_root, 'monodomain_results', 'analytical_case_enhanced')
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# Function Definitions
# =============================================================================

# Define the analytical solution function
def analytical_solution_v(x, y, t):
    """
    Analytical solution for the transmembrane potential v.

    Args:
        x (np.ndarray): x-coordinates.
        y (np.ndarray): y-coordinates.
        t (np.ndarray): Time points.

    Returns:
        np.ndarray: Analytical solution values.
    """
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

# Define source term function for PINNs
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


# Define source term function for FEM using UFL
def source_term_func(x, y, t):
    """
    Source term function for FEM.

    Args:
        x (ufl.Variable): x-coordinate.
        y (ufl.Variable): y-coordinate.
        t (ufl.Variable): Time coordinate.

    Returns:
        ufl.Expr: Source term expression.
    """
    return (
        8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.sin(t)
        + ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.cos(t)
    )

# =============================================================================
# FEM Simulation
# =============================================================================

# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 20, 20, 20  # Spatial and temporal resolution
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
    source_term_func=source_term_func,  # Updated function with 3 argumentsW
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

print("Solving second simulation with PINNs")

# Specify device
device = 'cpu'
print(f"Using device: {device}\n")

# Initialize the PINNs model with increased capacity and adjusted loss weights
model = EnhancedMonodomainSolverPINNs(
    num_inputs=3,          # For example, spatial coordinate x and time t
    num_layers=2,          # Number of hidden layers
    num_neurons=128,        # Number of neurons per hidden layer
    device='cpu',         
    source_term_func=source_term_func_pinns,
    M=1.0,                 # Parameter M in the monodomain equation
    use_ode=False,         # Whether to use ODE for current terms
    ode_func=None,         # ODE function if use_ode is True
    n_state_vars=0,        # Number of state variables if using ODE
    loss_function='L2',    # Type of loss function
    use_fourier=True,      # Enable Fourier Feature Embeddings
    fourier_dim=248,       # Number of Fourier features (m)
    sigma=3.0,             # Scale parameter for Fourier embeddings (within [1, 10])
    use_rwf=True,          # Enable Random Weight Factorization
    mu=1.0,                # Mean for RWF scale factors
    sigma_rwf=0.1          # Std dev for RWF scale factors
)

# =============================================================================
# Data Generation Using Latin Hypercube Sampling
# =============================================================================

from scipy.stats import qmc

# Define domain boundaries
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, T  # T is 1.0

# Define the number of points
N_collocation = 10000
N_ic = 1000
N_bc = 1000
N_val = 1000
N_test = 1000

# Define the number of validation points for collocation, IC, and BC
N_collocation_val = 1000
N_ic_val = 100
N_bc_val = 100

# =============================================================================
# Generate Collocation Points
# =============================================================================
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)

# Scale samples to the domain
X_collocation = sample.copy()
X_collocation[:, 0] = x_min + (x_max - x_min) * sample[:, 0]  # x
X_collocation[:, 1] = y_min + (y_max - y_min) * sample[:, 1]  # y
X_collocation[:, 2] = t_min + (t_max - t_min) * sample[:, 2]  # t

# Generate validation collocation points
sampler_val_collocation = qmc.LatinHypercube(d=3)
sample_val_collocation = sampler_val_collocation.random(n=N_collocation_val)

X_collocation_val = sample_val_collocation.copy()
X_collocation_val[:, 0] = x_min + (x_max - x_min) * sample_val_collocation[:, 0]  # x
X_collocation_val[:, 1] = y_min + (y_max - y_min) * sample_val_collocation[:, 1]  # y
X_collocation_val[:, 2] = t_min + (t_max - t_min) * sample_val_collocation[:, 2]  # t

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

# Compute expected initial condition values
expected_u0 = analytical_solution_v(X_ic[:, 0], X_ic[:, 1], X_ic[:, 2])
expected_u0 = expected_u0.reshape(-1, 1)

# Generate validation IC points
sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)

X_ic_val = sample_ic_val.copy()
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]  # x
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]  # y

# Add t=0
X_ic_val = np.hstack((X_ic_val, np.zeros((N_ic_val, 1))))

# Compute expected initial condition values for validation
expected_u0_val = analytical_solution_v(X_ic_val[:, 0], X_ic_val[:, 1], X_ic_val[:, 2])
expected_u0_val = expected_u0_val.reshape(-1, 1)

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
sampler_test = qmc.LatinHypercube(d=3)
sample_test = sampler_test.random(n=N_test)

# Scale samples to the domain
X_test = sample_test.copy()
X_test[:, 0] = x_min + (x_max - x_min) * sample_test[:, 0]  # x
X_test[:, 1] = y_min + (y_max - y_min) * sample_test[:, 1]  # y
X_test[:, 2] = t_min + (t_max - t_min) * sample_test[:, 2]  # t

# Compute analytical solution at test points
u_test = analytical_solution_v(X_test[:, 0], X_test[:, 1], X_test[:, 2])
u_test = u_test.reshape(-1, 1)

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
X_ic_norm = normalize_data(X_ic, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_norm = normalize_data(X_boundary, x_min, x_max, y_min, y_max, t_min, t_max)
X_collocation_val_norm = normalize_data(X_collocation_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_ic_val_norm = normalize_data(X_ic_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_val_norm = normalize_data(X_boundary_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_test_norm = normalize_data(X_test, x_min, x_max, y_min, y_max, t_min, t_max)

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

X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
u_test_tensor = torch.tensor(u_test, dtype=torch.float32).to(device)

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

import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define learning rate scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=500,
                                                 verbose=True)

# Define training parameters
epochs = 50000
batch_size = 16  # Set to None to use entire dataset

# Initialize variables for tracking progress
loss_list = []
val_loss_list = []
epoch_list = []
best_val_loss = float('inf')
no_improve_counter = 0
patience = 1000  # For early stopping
best_model_path = os.path.join(results_dir, 'best_model.pth')

# =============================================================================
# Training Loop with Validation Including IC and BC
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

# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = torch.mean((y_pred_test - u_test_tensor) ** 2)

print(f"Test Loss: {test_loss.item():.4e}")

# Compute error metrics
y_pred_test_np = y_pred_test.cpu().numpy()
u_test_np = u_test_tensor.cpu().numpy()

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_test = mean_squared_error(u_test_np, y_pred_test_np)
mae_test = mean_absolute_error(u_test_np, y_pred_test_np)
rmse_test = np.sqrt(mse_test)

print(f"Test MSE: {mse_test:.4e}")
print(f"Test MAE: {mae_test:.4e}")
print(f"Test RMSE: {rmse_test:.4e}")

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
# Compare Errors and Plotting Between PINNs and FEM using tricontourf
# =============================================================================

def plot_comparisons(model, solutions_fem, comparison_times, results_dir, device='cpu'):
    """
    Plot PINNs predictions, FEM solutions, analytical solutions, and absolute errors in a 2x2 subplot layout.

    Parameters:
        model (MonodomainSolverPINNs): Trained PINNs model.
        solutions_fem (dict): Dictionary with time keys mapping to FEM solutions.
        comparison_times (list): List of time points to compare.
        results_dir (str): Directory to save plots.
        device (str): Computational device ('cpu' or 'cuda').
    """
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Ensure that x_coords and y_coords are accessible within this function
    # They should be defined globally or passed as additional arguments
    # For this example, we'll assume they are accessible
    global x_coords, y_coords, x_min, x_max, y_min, y_max, t_min, t_max, normalize_data, analytical_solution_v

    # Create triangulation for tricontourf
    triang = Triangulation(x_coords, y_coords)

    for t in comparison_times:
        print(f"Comparing at time t = {t}")

        # Check if FEM solution exists for time t
        if t not in solutions_fem:
            print(f"FEM solution for time t = {t} not found.")
            continue

        # Extract FEM solution at time t
        fem_solution_t = solutions_fem[t]  # Assuming solutions_fem[t] is a numpy array of shape (N,)
        fem_v = fem_solution_t  # Shape: (N,)

        # Create input tensor for PINNs evaluation on the FEM mesh
        X_pinns_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))  # Shape: (N, 3)
        X_pinns_eval_norm = normalize_data(X_pinns_eval, x_min, x_max, y_min, y_max, t_min, t_max)  # Normalized
        X_pinns_eval_tensor = torch.tensor(X_pinns_eval_norm, dtype=torch.float32, device=device)  # Shape: (N, 3)

        # Enable gradient computation for validation (if needed in the validate method)
        # model.eval() is already called outside if necessary

        # Evaluate PINNs model
        with torch.set_grad_enabled(False):  # Disable gradients for faster computation
            y_pinns_pred = model.evaluate(X_pinns_eval_tensor)  # Shape: (N, 1)
            y_pinns_pred_np = y_pinns_pred.cpu().numpy().reshape(-1)  # Shape: (N,)

        # Compute analytical solution
        u_analytical = analytical_solution_v(x_coords, y_coords, t)  # Shape: (N,)

        # Compute absolute errors
        pinns_error = np.abs(y_pinns_pred_np - u_analytical)  # |PINNs - Analytical|
        fem_error = np.abs(fem_v - u_analytical)  # |FEM - Analytical|

        # Compute error metrics for PINNs
        mse_pinns = mean_squared_error(u_analytical, y_pinns_pred_np)
        mae_pinns = mean_absolute_error(u_analytical, y_pinns_pred_np)
        rmse_pinns = np.sqrt(mse_pinns)

        # Compute error metrics for FEM
        mse_fem = mean_squared_error(u_analytical, fem_v)
        mae_fem = mean_absolute_error(u_analytical, fem_v)
        rmse_fem = np.sqrt(mse_fem)

        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparisons at t = {t}', fontsize=16)

        # Subplot 1: PINNs Prediction
        cs1 = axs[0, 0].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
        cbar1 = fig.colorbar(cs1, ax=axs[0, 0])
        cbar1.set_label('PINNs Prediction')
        axs[0, 0].set_title('PINNs Prediction')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')

        # Subplot 2: PINNs Error
        cs2 = axs[0, 1].tricontourf(triang, pinns_error, levels=50, cmap='viridis')
        cbar2 = fig.colorbar(cs2, ax=axs[0, 1])
        cbar2.set_label('PINNs Error |PINNs - Analytical|')
        axs[0, 1].set_title('PINNs Absolute Error')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')

        # Subplot 3: FEM Prediction
        cs3 = axs[1, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
        cbar3 = fig.colorbar(cs3, ax=axs[1, 0])
        cbar3.set_label('FEM Solution')
        axs[1, 0].set_title('FEM Prediction')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')

        # Subplot 4: FEM Error
        cs4 = axs[1, 1].tricontourf(triang, fem_error, levels=50, cmap='viridis')
        cbar4 = fig.colorbar(cs4, ax=axs[1, 1])
        cbar4.set_label('FEM Error |FEM - Analytical|')
        axs[1, 1].set_title('FEM Absolute Error')
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
        print(f"  PINNs - MSE: {mse_pinns:.4e}, MAE: {mae_pinns:.4e}, RMSE: {rmse_pinns:.4e}")
        print(f"  FEM - MSE: {mse_fem:.4e}, MAE: {mae_fem:.4e}, RMSE: {rmse_fem:.4e}")
        print(f"Plots saved as {plot_filename}\n")

# Define comparison time points
comparison_times = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

# Call the plotting function
plot_comparisons(model, solutions_fem, comparison_times, results_dir, device=device)

print("Comparison between FEM and PINNs simulations complete.")


"""
Total Loss: 4.1398e-02, Validation Loss: 4.3766e-02 , 3 layers, 32 nodes, 128 batch , 30k epochs

    loss_function='L2',
    loss_weights={
        'pde_loss': 2.0,
        'IC_loss': 1.0,   # Increased weight for Initial Condition
        'BC_loss': 1.0,   # Increased weight for Boundary Condition
        'data_loss': 1.0,
        'ode_loss': 1.0
    }

Epoch 19200, PDE Loss: 1.4255e-02, IC Loss: 1.5311e-04, BC Loss: 8.1971e-03, ODE Loss: 0.0000e+00, Total Loss: 3.6861e-02, Validation Loss: 3.8159e-02

Model saved to /home/edvin/Desktop/master_edvin/monodomain_results/analytical_case/best_model.pth
New best model saved with validation loss 3.8159e-02

Total Loss: 4.1398e-02, Validation Loss: 4.3766e-02 , 3 layers, 64 nodes, 128 batch , 30k epochs

    loss_function='L2',
    loss_weights={
        'pde_loss': 2.0,
        'IC_loss': 1.0,   # Increased weight for Initial Condition
        'BC_loss': 1.0,   # Increased weight for Boundary Condition
        'data_loss': 1.0,
        'ode_loss': 1.0
    }


Epoch 20000, PDE Loss: 1.4667e-02, IC Loss: 1.0587e-03, BC Loss: 1.3741e-02, ODE Loss: 0.0000e+00, Total Loss: 5.8802e-02, Validation Loss: 8.1670e-02

model = MonodomainSolverPINNs(
    num_inputs=3,  # x, y, t
    num_layers=4,  # Increased layers for better approximation
    num_neurons=64,  # Increased neurons per layer
    device=device,
    source_term_func=source_term_func_pinns,
    M=M,
    use_ode=False,  # Assuming ODEs are not used; set to True if needed
    ode_func=None,  # Provide ODE function if use_ode=True
    n_state_vars=0,  # Number of state variables if use_ode=True
    loss_function='L2',
    loss_weights={
        'pde_loss': 3.0,
        'IC_loss': 1.0,   # Increased weight for Initial Condition
        'BC_loss': 1.0,   # Increased weight for Boundary Condition
        'data_loss': 1.0,
        'ode_loss': 1.0
    }
)




"""