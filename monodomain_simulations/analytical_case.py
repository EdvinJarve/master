# =============================================================================
# Imports
# =============================================================================

import sys
import os
import numpy as np
import time
import json
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
from utils.heart_solver_pinns import MonodomainSolverPINNs


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Define the results directory relative to the project root
results_dir = os.path.join(project_root, 'monodomain_results', 'analytical_case')
os.makedirs(results_dir, exist_ok=True)

# Define model parameters
model_params = {
    'num_inputs': 3,  # x, y, t
    'num_layers': 2,  # Number of hidden layers
    'num_neurons': 64, # Neurons per hidden layer
    'use_ode': False,  # Not using ODEs
    'n_state_vars': 0,  # No state variables
    'loss_function': 'L2',
    'weight_strategy': 'dynamic',  # Use dynamic weight adjustment
    'alpha': 1.0,  # Moving average parameter for weight updates
    'domain_bounds': {
        'x_min': 0.0,
        'x_max': 1.0,
        'y_min': 0.0,
        'y_max': 1.0,
        't_min': 0.0,
        't_max': 1.0  # T
    }
}

# Save model parameters to JSON
params_file = os.path.join(results_dir, 'model_parameters.json')
with open(params_file, 'w') as f:
    json.dump(model_params, f, indent=4)

print(f"Model parameters saved to: {params_file}")

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

# Initialize the PINNs model with parameters from config
model = MonodomainSolverPINNs(
    num_inputs=model_params['num_inputs'],
    num_layers=model_params['num_layers'],
    num_neurons=model_params['num_neurons'],
    device=device,
    source_term_func=source_term_func_pinns,
    M=M,
    use_ode=model_params['use_ode'],
    ode_func=None,
    n_state_vars=model_params['n_state_vars'],
    loss_function=model_params['loss_function'],
    weight_strategy=model_params['weight_strategy'],  # Set to 'dynamic'
    alpha=model_params['alpha'],  # Moving average parameter
    x_min=model_params['domain_bounds']['x_min'],
    x_max=model_params['domain_bounds']['x_max'],
    y_min=model_params['domain_bounds']['y_min'],
    y_max=model_params['domain_bounds']['y_max'],
    t_min=model_params['domain_bounds']['t_min'],
    t_max=model_params['domain_bounds']['t_max']
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
# Convert Data to PyTorch Tensors and Move to Device
# =============================================================================

# Convert to tensors using original (unscaled) data
X_collocation_tensor = torch.tensor(X_collocation, dtype=torch.float32).to(device)
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).to(device)
expected_u0_tensor = torch.tensor(expected_u0, dtype=torch.float32).to(device)
X_boundary_tensor = torch.tensor(X_boundary, dtype=torch.float32).to(device)
normal_vectors_tensor = torch.tensor(normal_vectors, dtype=torch.float32).to(device)

X_collocation_val_tensor = torch.tensor(X_collocation_val, dtype=torch.float32).to(device)
X_ic_val_tensor = torch.tensor(X_ic_val, dtype=torch.float32).to(device)
expected_u0_val_tensor = torch.tensor(expected_u0_val, dtype=torch.float32).to(device)
X_boundary_val_tensor = torch.tensor(X_boundary_val, dtype=torch.float32).to(device)
normal_vectors_val_tensor = torch.tensor(normal_vectors_val, dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
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
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Define training parameters
epochs = 20000
batch_size = 64  # Reduced from 64 to avoid index out of bounds
# Set to None to use entire dataset

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
    global x_coords, y_coords, x_min, x_max, y_min, y_max, t_min, t_max, analytical_solution_v

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
        X_pinns_eval_tensor = torch.tensor(X_pinns_eval, dtype=torch.float32, device=device)  # Shape: (N, 3)

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