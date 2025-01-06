import sys
import os
import numpy as np
import time
from dolfinx import fem, mesh
from mpi4py import MPI
import torch
import torch.nn as nn
import ufl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import qmc
import torch.optim as optim

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)

# Results directory
results_dir = os.path.join(project_root, 'monodomain_results', 'inverse_problem_final')
os.makedirs(results_dir, exist_ok=True)

# Problem Setup
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T

Nx, Ny, Nt = 100, 100, 100
dt = T / Nt
true_M = 1.0  # True conductivity used in the FEM simulation

def source_term_func(x, y, t):
    """
    Gaussian stimulus applied in upper left corner (x ∈ [0,0.2], y ∈ [0.8,1]) with temporal on/off windows.
    """
    x0 = 0.2
    y0 = 0.8
    sigma = 0.03
    t_on_start = 0.05
    t_on_end = 0.2
    t_off_start = 0.4
    t_off_end = 0.55

    gaussian = 50 * ufl.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
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
    Same Gaussian stimulus for PINNs as above.
    """
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    x0 = 0.2
    y0 = 0.8
    sigma = 0.03
    t_on_start = 0.05
    t_on_end = 0.2
    t_off_start = 0.4
    t_off_end = 0.55

    gaussian = 50 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    time_window = torch.zeros_like(t)
    ramp_up = (t >= t_on_start) & (t < t_on_end)
    time_window[ramp_up] = (t[ramp_up] - t_on_start) / (t_on_end - t_on_start)

    constant = (t >= t_on_end) & (t < t_off_start)
    time_window[constant] = 1.0

    ramp_down = (t >= t_off_start) & (t <= t_off_end)
    time_window[ramp_down] = 1 - (t[ramp_down] - t_off_start) / (t_off_end - t_off_start)

    return gaussian * time_window

# FEM Simulation
fem_data_file = os.path.join(results_dir, 'fem_data.npz')
if not os.path.exists(fem_data_file):
    print("Running FEM simulation...")
    domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
    sim_fem = MonodomainSolverFEM(
        mesh=domain_mesh,
        T=T,
        dt=dt,
        M_i=true_M,
        source_term_func=source_term_func,
        initial_v=0.0
    )

    time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    start_time_fem = time.time()
    errors_fem, computation_time_fem, solutions_fem = sim_fem.run(time_points=time_points)
    end_time_fem = time.time()
    print(f"FEM simulation complete in {end_time_fem - start_time_fem:.2f} seconds")

    dof_coords = sim_fem.V.tabulate_dof_coordinates()
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]

    # Convert time_points to a NumPy array
    time_points_array = np.array(time_points)
    fem_matrix = np.vstack([solutions_fem[t] for t in time_points]).T
    np.savez(fem_data_file, x_coords=x_coords, y_coords=y_coords, time_points=time_points_array, fem_solutions=fem_matrix)
    print(f"FEM data saved to {fem_data_file}")
else:
    fem_data = np.load(fem_data_file)
    x_coords = fem_data['x_coords']
    y_coords = fem_data['y_coords']
    time_points = fem_data['time_points']
    fem_matrix = fem_data['fem_solutions']

# Construct solutions_fem dict
solutions_fem = {t: fem_matrix[:, i] for i, t in enumerate(time_points)}
triang = Triangulation(x_coords, y_coords)

# PINN Hyperparams for Inverse Problem
hyperparams = {
    'num_inputs': 3,
    'num_layers': 2,
    'num_neurons': 256,
    'device': 'cpu',
    'initial_M': 0.0,  
    'learning_rate': 1e-3,
    'epochs': 10000,
    'batch_size': 128,
    'loss_weights': None,  # Not needed for dynamic weights
    'weight_strategy': 'dynamic',  # enable dynamic weights
    'alpha': 0.9
}
# Generate training data
X_ic = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))
# We don't have an analytical solution for this specific source term, 
# but if we assume initial_v=0.0, expected_u0 is zero:
expected_u0 = np.zeros((len(x_coords), 1))

# Let's pick a few time points for data sampling from FEM to help identify M
chosen_times = [0.1, 0.2, 0.4, 0.8]
X_data = []
v_data = []
for t in chosen_times:
    idx = list(time_points).index(t)
    X_t = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))
    v_t = fem_matrix[:, idx]
    X_data.append(X_t)
    v_data.append(v_t)
X_data = np.vstack(X_data)
v_data = np.hstack(v_data)

# Collocation points
N_collocation = 20000
X_collocation = np.random.uniform([x_min, y_min, t_min], [x_max, y_max, t_max], size=(N_collocation, 3))

# Boundary points
N_boundary = 4000
def generate_boundary_points(num_points):
    points = []
    normals = []
    for _ in range(num_points):
        tt = np.random.uniform(t_min, t_max)
        side = np.random.choice(['x_min', 'x_max', 'y_min', 'y_max'])
        if side == 'x_min':
            xx = x_min
            yy = np.random.uniform(y_min, y_max)
            n = [-1, 0]
        elif side == 'x_max':
            xx = x_max
            yy = np.random.uniform(y_min, y_max)
            n = [1, 0]
        elif side == 'y_min':
            xx = np.random.uniform(x_min, x_max)
            yy = y_min
            n = [0, -1]
        else:  # y_max
            xx = np.random.uniform(x_min, x_max)
            yy = y_max
            n = [0, 1]
        points.append([xx, yy, tt])
        normals.append(n)
    return np.array(points), np.array(normals)

X_boundary, normal_vectors = generate_boundary_points(N_boundary)

# Initialize Inverse PINN
pinn = InverseMonodomainSolverPINNs(
    num_inputs=hyperparams['num_inputs'],
    num_layers=hyperparams['num_layers'],
    num_neurons=hyperparams['num_neurons'],
    device=hyperparams['device'],
    source_term_func=source_term_func_pinns,
    initial_M=hyperparams['initial_M'],
    loss_weights=None,     # If using dynamic weights, loss_weights can be None or ignored
    weight_strategy='dynamic',  # Enable dynamic weight updates
    alpha=0.9,             # Moving average parameter for dynamic weights (adjust if needed)
)

# Convert all data to tensors
X_collocation_tensor = torch.tensor(X_collocation, dtype=torch.float32).to(hyperparams['device'])
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).to(hyperparams['device'])
expected_u0_tensor = torch.tensor(expected_u0, dtype=torch.float32).to(hyperparams['device'])
X_boundary_tensor = torch.tensor(X_boundary, dtype=torch.float32).to(hyperparams['device'])
normal_vectors_tensor = torch.tensor(normal_vectors, dtype=torch.float32).to(hyperparams['device'])

X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(hyperparams['device'])
v_data_tensor = torch.tensor(v_data, dtype=torch.float32).unsqueeze(-1).to(hyperparams['device'])

# Assign data to model
pinn.X_collocation = X_collocation_tensor
pinn.X_ic = X_ic_tensor
pinn.expected_u0 = expected_u0_tensor
pinn.X_boundary = X_boundary_tensor
pinn.normal_vectors = normal_vectors_tensor
pinn.X_data = X_data_tensor
pinn.expected_data = v_data_tensor

optimizer = optim.Adam(pinn.parameters(), lr=hyperparams['learning_rate'])

# Training Loop
epochs = hyperparams['epochs']
batch_size = hyperparams['batch_size']
M_estimates = []
best_val_loss = float('inf')
no_improve_counter = 0
patience = 1000
best_model_path = os.path.join(results_dir, 'best_model.pth')

for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, data_loss_val, ode_loss, total_loss = pinn.train_step(optimizer, batch_size)
    
    # For now, we won't do a separate validation step unless we create val data.
    # Just track training loss and M estimates.
    if epoch % 100 == 0:
        M_current = pinn.M.item()
        M_estimates.append(M_current)
        print(f"Epoch {epoch}, Total Loss: {total_loss:.4e}, PDE: {pde_loss:.4e}, IC: {IC_loss:.4e}, BC: {BC_loss:.4e}, Data: {data_loss_val:.4e}, M: {M_current:.4e}")
        
        # Save best model based on total_loss as a proxy (no validation data)
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            no_improve_counter = 0
            pinn.save_model(best_model_path)
            print(f"New best model saved with loss {best_val_loss:.4e}")
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print("Early stopping triggered.")
                break

# Save M predictions
M_predictions_file = os.path.join(results_dir, 'M_predictions.txt')
np.savetxt(M_predictions_file, M_estimates, fmt='%.6e')
print(f"M predictions saved to {M_predictions_file}")

# Compute final M errors
M_final = M_estimates[-1] if M_estimates else pinn.M.item()
M_abs_error = abs(M_final - true_M)
M_rel_error = M_abs_error / abs(true_M)

M_abs_error_file = os.path.join(results_dir, 'M_abs_error.txt')
with open(M_abs_error_file, 'w') as f:
    f.write(f"{M_abs_error:.6e}\n")
print(f"M absolute error saved to {M_abs_error_file}")

M_rel_error_file = os.path.join(results_dir, 'M_rel_error.txt')
with open(M_rel_error_file, 'w') as f:
    f.write(f"{M_rel_error:.6e}\n")
print(f"M relative error saved to {M_rel_error_file}")

# Save hyperparams
hyperparams_file = os.path.join(results_dir, 'hyperparams.json')
with open(hyperparams_file, 'w') as f:
    json.dump(hyperparams, f, indent=4)
print(f"Hyperparameters saved to {hyperparams_file}")

pinn.load_model(best_model_path)
pinn.eval()

# Comparison times
comparison_times = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

def plot_comparisons(model, solutions_fem, comparison_times, results_dir, device='cpu'):
    """
    Plot PINNs predictions, FEM solutions, and their absolute errors in a 2x2 subplot layout.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    os.makedirs(results_dir, exist_ok=True)
    triang = Triangulation(x_coords, y_coords)

    model.eval()
    with torch.no_grad():
        for t in comparison_times:
            print(f"Comparing at time t = {t}")
            if t not in solutions_fem:
                print(f"FEM solution for time t={t} not found. Skipping.")
                continue
            fem_v = solutions_fem[t]
            X_pinns_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))
            
            # Normalize if needed (if the model expected normalized inputs)
            # If no scaling_func is used in the inverse model, just pass directly
            # If scaling was used, apply it here as well
            
            X_pinns_eval_tensor = torch.tensor(X_pinns_eval, dtype=torch.float32, device=device)
            y_pinns_pred = model.evaluate(X_pinns_eval_tensor)
            y_pinns_pred_np = y_pinns_pred.cpu().numpy().reshape(-1)

            error = np.abs(y_pinns_pred_np - fem_v)
            mse = mean_squared_error(fem_v, y_pinns_pred_np)
            mae = mean_absolute_error(fem_v, y_pinns_pred_np)
            rmse = np.sqrt(mse)

            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Comparisons at t = {t}', fontsize=16)

            # FEM Prediction
            cs1 = axs[0, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
            fig.colorbar(cs1, ax=axs[0, 0]).set_label('FEM Prediction')
            axs[0, 0].set_title('FEM Prediction')
            axs[0, 0].set_xlabel('x')
            axs[0, 0].set_ylabel('y')

            # PINNs Prediction
            cs2 = axs[0, 1].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
            fig.colorbar(cs2, ax=axs[0, 1]).set_label('PINNs Prediction')
            axs[0, 1].set_title('PINNs Prediction')
            axs[0, 1].set_xlabel('x')
            axs[0, 1].set_ylabel('y')

            # Absolute Error
            cs3 = axs[1, 0].tricontourf(triang, error, levels=50, cmap='viridis')
            fig.colorbar(cs3, ax=axs[1, 0]).set_label('Absolute Error |PINNs - FEM|')
            axs[1, 0].set_title('Absolute Error')
            axs[1, 0].set_xlabel('x')
            axs[1, 0].set_ylabel('y')

            # Re-plot FEM or source term if desired. Here just FEM again.
            cs4 = axs[1, 1].tricontourf(triang, fem_v, levels=50, cmap='viridis')
            fig.colorbar(cs4, ax=axs[1, 1]).set_label('FEM Solution')
            axs[1, 1].set_title('FEM again or other field')
            axs[1, 1].set_xlabel('x')
            axs[1, 1].set_ylabel('y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename = f'comparison_t_{t}.png'
            plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
            plt.close()
            print(f"Plots saved for time t = {t}")
            print(f"Time {t}: PINNs - MSE: {mse:.4e}, MAE: {mae:.4e}, RMSE: {rmse:.4e}")
            print(f"Plots saved as {plot_filename}\n")

plot_comparisons(pinn, solutions_fem, comparison_times, results_dir, device=hyperparams['device'])

print("Comparison between FEM and PINNs simulations complete.")
