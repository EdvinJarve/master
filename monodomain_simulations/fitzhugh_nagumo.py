# =============================================================================
# 1. Imports and Backend Configuration
# =============================================================================

import sys
import os
import numpy as np
import time
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
import imageio  # For creating GIF
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from matplotlib.tri import Triangulation

# Set the Matplotlib backend to 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # Now safe to import pyplot

# =============================================================================
# 2. Project Setup (Paths, Directories)
# =============================================================================

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the updated MonodomainSolverFEM class
from utils.heart_solver_fem import MonodomainSolverFEM, MonodomainSolverFEMHardcode
from utils.heart_solver_pinns import MonodomainSolverPINNs

# Ensure directory for figures exists
results_dir = os.path.join(project_root, 'monodomain_results', 'fitzhugh_nagumo')
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# 3. Function Definitions
# =============================================================================

# Define the FitzHugh-Nagumo ODE system for PINNs
def ode_func_fhn(v, state_vars, X):
    """
    Compute the ODE residuals for the FitzHugh-Nagumo model.

    Parameters:
        v (torch.Tensor): Membrane potential tensor of shape (N, 1).
        state_vars (torch.Tensor): State variable tensor of shape (N, n_state_vars).
        X (torch.Tensor): Input coordinates tensor of shape (N, D).

    Returns:
        torch.Tensor: ODE residuals tensor of shape (N, 2).
    """
    # Extract state variable w
    w = state_vars[:, 0:1]  # Assuming w is the first state variable

    # Parameters
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0

    # Time coordinate
    t = X[:, -1:]  # Shape: (N, 1)

    # Applied current i_app (time-dependent)f
    #i_app = torch.where(t <= 0.3, 0.8, 0.0)

    # ODE for v
    dv_dt = c1 * v * (v - a) * (1 - v) - c2 * v * w 

    # ODE for w
    dw_dt = b * (v - c3 * w)

    # ODE residuals
    ode_residual = torch.cat([dv_dt, dw_dt], dim=1)  # Shape: (N, 2)

    return ode_residual


# =============================================================================
# 3. Function Definitions
# =============================================================================
def ode_system_fhn(t, y):
    """
    FitzHugh-Nagumo ODE system for v and w.

    Parameters:
        t (float): Current time.
        y (np.ndarray): Flattened array containing [v, w], shape (2 * num_nodes,)

    Returns:
        dy_dt (np.ndarray): Flattened array containing [dv_dt, dw_dt], shape (2 * num_nodes,)
    """
    num_nodes = y.size // 2
    v = y[:num_nodes]
    w = y[num_nodes:]

    # Parameters
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0

    # Compute dv/dt and dw/dt
    dv_dt = c1 * v * (v - a) * (1 - v) - c2 * v * w
    dw_dt = b * (v - c3 * w)

    dy_dt = np.concatenate([dv_dt, dw_dt])
    return dy_dt

def initial_v_function(x):
    """
    Initial membrane potential v.
    
    Parameters:
        x (np.ndarray): Array of coordinates, shape (2, num_nodes).
    
    Returns:
        np.ndarray: Initial v values, shape (num_nodes,).
    """
    v_initial = np.zeros_like(x[0])
    v_initial[x[0] <= 1/3] = 0.8  # Set v = 0.8 in the left third of the domain
    return v_initial

# Define the initial condition for s (w)
def initial_w_function(x):
    """
    Initial state variable w.
    
    Parameters:
        x (np.ndarray): Array of coordinates, shape (2, num_nodes).
    
    Returns:
        np.ndarray: Initial w values, shape (num_nodes,).
    """
    return np.zeros_like(x[0])  # For example, w = 0 everywhere

# =============================================================================
# 4. Simulation Parameters and Mesh Creation
# =============================================================================

# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 100, 100, 100  # Mesh resolution and number of time steps
T = 1.0  # Total simulation time
dt = T / Nt  # Time step size
M = 1.0  # Diffusion coefficient
theta = 0.5  # Splitting parameter for Strang splitting

# Create mesh
domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 0.99]

# =============================================================================
# 5. Initialize and Run the FEM Simulation
# =============================================================================

# Initialize and run the FEM simulation with the ODE system and initial conditions
sim_fem = MonodomainSolverFEMHardcode(
    mesh=domain_mesh,
    T=T,
    dt=dt,
    M_i=M,
    source_term_func=None,          # No explicit source term
    ode_system=ode_system_fhn,      # ODE system for FHN
    initial_v=initial_v_function,   # Initial membrane potential function
    initial_s=initial_w_function,   # Initial state variable function
    theta=theta                     # Splitting parameter
)

# Run the simulation
errors_v, computation_time, solutions_fem = sim_fem.run(
    analytical_solution_v=None,  # Replace with analytical solution function if available
    time_points=time_points             # Replace with specific time points if needed
)

dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]
print(f"Simulation completed in {computation_time:.2f} seconds.")

# =============================================================================
# 4. PINNs Simulation Setup
# =============================================================================

# Specify device
device = 'cpu'
print(f"Using device: {device}\n")

# Define domain boundaries
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Define the number of points
N_collocation = 20000
N_ic = 4000
N_bc = 4000
N_val = 2000

# Define the number of validation points for collocation, IC, and BC
N_collocation_val = 1000
N_ic_val = 100
N_bc_val = 100

# Create collocation points using Latin Hypercube Sampling
def sample_lhs(n_samples, bounds):
    """
    Sample n_samples using Latin Hypercube Sampling within the specified bounds.
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n=n_samples)
    # Scale samples to the bounds
    X = np.empty_like(sample)
    for i, (lower, upper) in enumerate(bounds):
        X[:, i] = lower + (upper - lower) * sample[:, i]
    return X

# Collocation points in the domain
collocation_bounds = [(x_min, x_max), (y_min, y_max), (t_min, t_max)]
X_collocation = sample_lhs(N_collocation, collocation_bounds)

# Validation collocation points
X_collocation_val = sample_lhs(N_collocation_val, collocation_bounds)

# Initial condition points
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)
X_ic = np.zeros((N_ic, 3))
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]  # x
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]  # y
X_ic[:, 2] = t_min  # t = 0

# Set expected initial condition values
v_initial_values = initial_v_function([X_ic[:, 0], X_ic[:, 1]])
expected_u0 = np.zeros((N_ic, 2))  # For v and w
expected_u0[:, 0] = v_initial_values  # v initial
expected_u0[:, 1] = 0.0  # w initial

# Validation initial condition points
sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)
X_ic_val = np.zeros((N_ic_val, 3))
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]  # x
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]  # y
X_ic_val[:, 2] = t_min  # t = 0

# Set expected initial condition values for validation
v_initial_values_val = initial_v_function([X_ic_val[:, 0], X_ic_val[:, 1]])
expected_u0_val = np.zeros((N_ic_val, 2))  # For v and w
expected_u0_val[:, 0] = v_initial_values_val  # v initial
expected_u0_val[:, 1] = 0.0  # w initial

# Boundary condition points
N_per_boundary = N_bc // 4

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

# Validation boundary condition points
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
# Normalize Data
# =============================================================================

def normalize_data(X, x_min, x_max, y_min, y_max, t_min, t_max):
    """
    Normalize spatial and temporal coordinates to [0, 1].
    """
    X_norm = X.copy()
    X_norm[:, 0] = (X[:, 0] - x_min) / (x_max - x_min)  # x
    X_norm[:, 1] = (X[:, 1] - y_min) / (y_max - y_min)  # y
    X_norm[:, 2] = (X[:, 2] - t_min) / (t_max - t_min)  # t
    return X_norm

# Apply normalization
X_collocation_norm = normalize_data(X_collocation, x_min, x_max, y_min, y_max, t_min, t_max)
X_collocation_val_norm = normalize_data(X_collocation_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_ic_norm = normalize_data(X_ic, x_min, x_max, y_min, y_max, t_min, t_max)
X_ic_val_norm = normalize_data(X_ic_val, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_norm = normalize_data(X_boundary, x_min, x_max, y_min, y_max, t_min, t_max)
X_boundary_val_norm = normalize_data(X_boundary_val, x_min, x_max, y_min, y_max, t_min, t_max)

# =============================================================================
# Convert Data to PyTorch Tensors and Move to Device
# =============================================================================

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

# =============================================================================
# Initialize the PINNs Model
# =============================================================================

n_state_vars = 1  # Number of state variables (w)

model = MonodomainSolverPINNs(
    num_inputs=3,  # x, y, t
    num_layers=2,
    num_neurons=512,
    device=device,
    source_term_func=lambda x, t: torch.zeros_like(t),  # No explicit source term
    M=1.0,  # Conductivity
    use_ode=True,
    ode_func=ode_func_fhn,
    n_state_vars=n_state_vars,
    loss_function='L2',
    loss_weights={
        'pde_loss': 1.0,
        'IC_loss': 1.0,
        'BC_loss': 1.0,
        'data_loss': 1.0,
        'ode_loss': 1.0
    }
)

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
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define learning rate scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=500,
                                                 verbose=True)

# Define training parameters
epochs = 15000
batch_size = 256  # Adjust based on your system's memory

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
              f'Total Loss: {total_loss:.4e}, Validation Loss: {total_val_loss:.4e}')
        
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

time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]  # Must match FEM time points

# Collect PINNs predictions
solutions_pinns = {}

# Generate grid points matching the PINNs grid
Nx_pinns, Ny_pinns = 100, 100
x_lin = np.linspace(0.0, 1.0, Nx_pinns)  # Assuming x_min=0.0, x_max=1.0
y_lin = np.linspace(0.0, 1.0, Ny_pinns)  # Assuming y_min=0.0, y_max=1.0
x_grid, y_grid = np.meshgrid(x_lin, y_lin)
x_flat = x_grid.flatten()  # Shape: (10000,)
y_flat = y_grid.flatten()  # Shape: (10000,)

for idx, t in enumerate(time_points):
    # Create input tensor with current time
    X_pinns_eval = np.column_stack((x_flat, y_flat, np.full_like(x_flat, t)))  # Shape: (10000, 3)
    X_pinns_eval_norm = normalize_data(X_pinns_eval, 0.0, 1.0, 0.0, 1.0, 0.0, T)  # Replace with actual normalization
    X_pinns_eval_tensor = torch.tensor(X_pinns_eval_norm, dtype=torch.float32).to(device)
    
    # Evaluate PINNs model
    with torch.no_grad():
        outputs = model.evaluate(X_pinns_eval_tensor)
        v_pred = outputs[:, 0].cpu().numpy().reshape(-1)  # Shape: (10000,)
    
    # Store the predictions
    solutions_pinns[t] = v_pred

# =============================================================================
# Interpolate FEM Solutions onto the PINNs Grid
# =============================================================================

solutions_fem_interp = {}

for t in time_points:
    if t not in solutions_fem:
        print(f"Warning: FEM solution at time t = {t} not found. Skipping interpolation.")
        continue
    
    # Get the FEM solution at time t (shape: (441,))
    v_fem = solutions_fem[t]  # Shape: (441,)
    
    # Combine x and y coordinates into a single array of shape (441, 2)
    points = np.column_stack((x_coords, y_coords))  # Shape: (441, 2)
    
    # Combine x_flat and y_flat into a single array of shape (10000, 2)
    xi = np.column_stack((x_flat, y_flat))          # Shape: (10000, 2)
    
    # Perform linear interpolation onto the PINNs grid (x_flat, y_flat)
    v_fem_interp = griddata(
        points=points,  # Original FEM mesh coordinates, shape: (441, 2)
        values=v_fem,    # FEM solution values, shape: (441,)
        xi=xi,           # PINNs grid coordinates, shape: (10000, 2)
        method='linear'  # Interpolation method
    )
    
    # Handle possible NaNs resulting from interpolation
    nan_indices = np.isnan(v_fem_interp)
    if np.any(nan_indices):
        # Fill NaNs using nearest-neighbor interpolation
        v_fem_interp[nan_indices] = griddata(
            points=points,
            values=v_fem,
            xi=xi[nan_indices],
            method='nearest'
        )
    
    # Store the interpolated FEM solution
    solutions_fem_interp[t] = v_fem_interp

# =============================================================================
# Compare with PINNs Simulation
# =============================================================================

def plot_comparisons(t_array, x_flat, y_flat, solutions_fem_interp, solutions_pinns, results_dir):
    """
    Plot comparison between FEM and PINNs solutions and their absolute error for multiple time points.

    Parameters:
        t_array (list or np.ndarray): Array of time points for the plots.
        x_flat (np.ndarray): x-coordinates of the PINNs grid points.
        y_flat (np.ndarray): y-coordinates of the PINNs grid points.
        solutions_fem_interp (dict): Dictionary of interpolated FEM solutions with time keys.
        solutions_pinns (dict): Dictionary of PINNs solutions with time keys.
        results_dir (str): Directory to save the plots.
    """
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create triangulation for plotting using PINNs grid
    triang = Triangulation(x_flat, y_flat)

    for t in t_array:
        print(f"Creating comparison plot for time t = {t}")

        # Check if both solutions exist
        if t not in solutions_fem_interp:
            print(f"FEM interpolated solution at t = {t} not found. Skipping plot.")
            continue
        if t not in solutions_pinns:
            print(f"PINNs solution at t = {t} not found. Skipping plot.")
            continue

        # Get FEM and PINNs solutions at time t
        v_fem = solutions_fem_interp[t]  # Shape: (10000,)
        v_pinns = solutions_pinns[t]     # Shape: (10000,)
        error = np.abs(v_fem - v_pinns)  # Shape: (10000,)

        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparison at t = {t}', fontsize=16)

        # Subplot 1: FEM Prediction
        cs1 = axs[0, 0].tricontourf(triang, v_fem, levels=50, cmap='viridis')
        fig.colorbar(cs1, ax=axs[0, 0], label='FEM Prediction')
        axs[0, 0].set_title('FEM Prediction')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')

        # Subplot 2: PINNs Prediction
        cs2 = axs[0, 1].tricontourf(triang, v_pinns, levels=50, cmap='viridis')
        fig.colorbar(cs2, ax=axs[0, 1], label='PINNs Prediction')
        axs[0, 1].set_title('PINNs Prediction')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')

        # Subplot 3: Absolute Error (|PINNs - FEM|)
        cs3 = axs[1, 0].tricontourf(triang, error, levels=50, cmap='viridis')
        fig.colorbar(cs3, ax=axs[1, 0], label='Absolute Error |PINNs - FEM|')
        axs[1, 0].set_title('Absolute Error')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')

        # Subplot 4: Empty or customizable plot area
        axs[1, 1].axis('off')  # Leave it empty or add another visualization as needed

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'comparison_t_{t:.2f}.png'
        plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
        plt.close()
        print(f"Comparison plot saved as {plot_filename}")

# =============================================================================
# Calculate and Plot Errors at Selected Time Points
# =============================================================================

selected_times = [0.2, 0.4, 0.8, 1.0]  # Same as t_array
error_list = []

for t in selected_times:
    if t not in solutions_fem_interp:
        print(f"FEM interpolated solution at t = {t} not found. Skipping error calculation.")
        continue
    if t not in solutions_pinns:
        print(f"PINNs solution at t = {t} not found. Skipping error calculation.")
        continue

    v_pinns = solutions_pinns[t]      # Shape: (10000,)
    v_fem = solutions_fem_interp[t]   # Shape: (10000,)
    error = np.abs(v_pinns - v_fem)   # Shape: (10000,)
    mse = np.mean(error**2)           # Mean Squared Error
    error_list.append((t, mse))
    print(f"Time {t}: MSE between PINNs and FEM: {mse:.4e}")

# =============================================================================
# Plot Error Over Time
# =============================================================================
if error_list:
    times, mse_errors = zip(*error_list)
    plt.figure(figsize=(10, 6))
    plt.plot(times, mse_errors, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Error Between PINNs and FEM Over Time')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'error_plot.png'), dpi=300)
    plt.close()
    print("Error plot saved as 'error_plot.png'.")
else:
    print("No errors to plot. Ensure that `selected_times` are correctly defined and present in both solutions.")

plt.figure(figsize=(10, 6))
plt.plot(epoch_list, loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_plot.png'), dpi=300)
plt.close()
print("Loss plot saved.")

plot_comparisons(time_points, x_flat, y_flat, solutions_fem_interp, solutions_pinns, results_dir)

print("Comparison between FEM and PINNs simulations complete.")

'''
# =============================================================================
# Create GIFs for Both Simulations
# =============================================================================

print("Creating GIFs of propagation...")

# Create a list to store filenames of saved images
image_filenames_pinns = []
image_filenames_fem = []

# Normalize color scale based on the maximum and minimum values across all time steps
v_values_pinns = np.array(list(solutions_pinns.values()))
v_values_fem = np.array(list(solutions_fem_interp.values()))
vmin = min(np.min(v_values_pinns), np.min(v_values_fem))
vmax = max(np.max(v_values_pinns), np.max(v_values_fem))

# Loop over time points and create plots
for idx, t in enumerate(time_points_pinns):
    v_pinns_array = solutions_pinns[t]
    v_fem_array = solutions_fem_interp[t]
    
    # Create figure for PINNs
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x_flat, y_flat, v_pinns_array, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Membrane Potential (v)')
    plt.title(f'PINNs - Time: {t:.3f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    filename_pinns = os.path.join(results_dir, f'frame_pinns_{idx:04d}.png')
    plt.savefig(filename_pinns, dpi=100)
    plt.close()
    image_filenames_pinns.append(filename_pinns)
    
    # Create figure for FEM
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x_flat, y_flat, v_fem_array, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Membrane Potential (v)')
    plt.title(f'FEM - Time: {t:.3f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    filename_fem = os.path.join(results_dir, f'frame_fem_{idx:04d}.png')
    plt.savefig(filename_fem, dpi=100)
    plt.close()
    image_filenames_fem.append(filename_fem)

# Create GIFs using imageio
gif_filename_pinns = os.path.join(results_dir, 'propagation_pinns.gif')
gif_filename_fem = os.path.join(results_dir, 'propagation_fem.gif')
with imageio.get_writer(gif_filename_pinns, mode='I', duration=0.05) as writer_pinns:
    for filename in image_filenames_pinns:
        image = imageio.imread(filename)
        writer_pinns.append_data(image)

with imageio.get_writer(gif_filename_fem, mode='I', duration=0.05) as writer_fem:
    for filename in image_filenames_fem:
        image = imageio.imread(filename)
        writer_fem.append_data(image)

# Optionally, delete the individual frame images
for filename in image_filenames_pinns + image_filenames_fem:
    os.remove(filename)

print(f"GIFs saved as {gif_filename_pinns} and {gif_filename_fem}")

# =============================================================================
# Create Side-by-Side GIF
# =============================================================================

print("Creating side-by-side GIF...")

# Ensure the number of frames is the same
assert len(image_filenames_pinns) == len(image_filenames_fem), "Mismatch in number of frames"

side_by_side_filenames = []
for idx in range(len(image_filenames_pinns)):
    # Read images
    img_pinns = imageio.imread(image_filenames_pinns[idx])
    img_fem = imageio.imread(image_filenames_fem[idx])
    
    # Combine images side by side
    combined_img = np.hstack((img_pinns, img_fem))
    
    # Save combined image
    combined_filename = os.path.join(results_dir, f'frame_combined_{idx:04d}.png')
    imageio.imwrite(combined_filename, combined_img)
    side_by_side_filenames.append(combined_filename)

# Create GIF
gif_filename_combined = os.path.join(results_dir, 'propagation_combined.gif')
with imageio.get_writer(gif_filename_combined, mode='I', duration=0.05) as writer_combined:
    for filename in side_by_side_filenames:
        image = imageio.imread(filename)
        writer_combined.append_data(image)

# Optionally, delete the individual combined images
for filename in side_by_side_filenames:
    os.remove(filename)

print(f"Combined GIF saved as {gif_filename_combined}")
'''