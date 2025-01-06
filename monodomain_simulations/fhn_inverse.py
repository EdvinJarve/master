import sys
import os
import numpy as np
import time
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
import torch
import torch.nn as nn
import torch.optim as optim
import json
from matplotlib.tri import Triangulation
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEMHardcode
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)

results_dir = os.path.join(project_root, 'monodomain_results', 'fhn_inverse')
os.makedirs(results_dir, exist_ok=True)

# Problem Setup
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T
Nx, Ny, Nt = 100, 100, 100
dt = T / Nt
true_M = 1.0
theta = 0.5

def ode_func_fhn(v, state_vars, X):
    # FitzHugh-Nagumo parameters
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0

    w = state_vars[:, 0:1]
    dv_dt = c1 * v * (v - a) * (1 - v) - c2 * v * w
    dw_dt = b * (v - c3 * w)
    ode_residual = torch.cat([dv_dt, dw_dt], dim=1)
    return ode_residual

def ode_system_fhn(t, y):
    num_nodes = y.size // 2
    v = y[:num_nodes]
    w = y[num_nodes:]

    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0

    dv_dt = c1 * v * (v - a) * (1 - v) - c2 * v * w
    dw_dt = b * (v - c3 * w)
    return np.concatenate([dv_dt, dw_dt])

def initial_v_function(x):
    v_initial = np.zeros_like(x[0])
    v_initial[x[0] <= 1/3] = 0.8
    return v_initial

def initial_w_function(x):
    return np.zeros_like(x[0])

# FEM Simulation
domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 0.99]

sim_fem = MonodomainSolverFEMHardcode(
    mesh=domain_mesh,
    T=T,
    dt=dt,
    M_i=true_M,
    source_term_func=None,
    ode_system=ode_system_fhn,
    initial_v=initial_v_function,
    initial_s=initial_w_function,
    theta=theta
)

errors_v, computation_time, solutions_fem = sim_fem.run(
    analytical_solution_v=None,
    time_points=time_points
)

dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

print(f"FEM simulation completed in {computation_time:.2f} seconds.")

# Construct solutions_fem dict {t: v_values}
solutions_fem_dict = {t: solutions_fem[t] for t in time_points}

triang = Triangulation(x_coords, y_coords)

# PINN Hyperparams for inverse problem
hyperparams = {
    'num_inputs': 3,      # x, y, t
    'num_layers': 2,
    'num_neurons': 256,
    'device': 'cpu',
    'initial_M': 1.0,
    'learning_rate': 1e-3,
    'epochs': 2000,
    'batch_size': 64,
    'weight_strategy': 'dynamic',  # Enable dynamic weights
    'alpha': 0.9,
    'use_ode': True,
    'n_state_vars': 1    # We have w as one state variable
}

# Generate training data for PINN
# IC data (t=0, v known from initial_v_function)
X_ic = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))
v_ic = initial_v_function((x_coords, y_coords)).reshape(-1, 1)

# Data from FEM at a few time points to help identify M
chosen_times = [0.1, 0.2, 0.4, 0.8]
X_data = []
v_data = []
for t in chosen_times:
    if t not in solutions_fem_dict:
        continue
    v_t = solutions_fem_dict[t]
    X_t = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t)))
    X_data.append(X_t)
    v_data.append(v_t)
X_data = np.vstack(X_data)
v_data = np.hstack(v_data).reshape(-1,1)

# Collocation points
N_collocation = 10000
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
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs

pinn = InverseMonodomainSolverPINNs(
    num_inputs=hyperparams['num_inputs'],
    num_layers=hyperparams['num_layers'],
    num_neurons=hyperparams['num_neurons'],
    device=hyperparams['device'],
    source_term_func=None,  # No explicit source, ODE handles dynamics
    initial_M=hyperparams['initial_M'],
    use_ode=True,
    ode_func=ode_func_fhn,
    n_state_vars=hyperparams['n_state_vars'],
    loss_function='L2',
    loss_weights=None,
    weight_strategy=hyperparams['weight_strategy'],
    alpha=hyperparams['alpha']
)

X_collocation_tensor = torch.tensor(X_collocation, dtype=torch.float32).to(hyperparams['device'])
X_ic_tensor = torch.tensor(X_ic, dtype=torch.float32).to(hyperparams['device'])
v_ic_tensor = torch.tensor(v_ic, dtype=torch.float32).to(hyperparams['device'])
X_boundary_tensor = torch.tensor(X_boundary, dtype=torch.float32).to(hyperparams['device'])
normal_vectors_tensor = torch.tensor(normal_vectors, dtype=torch.float32).to(hyperparams['device'])
X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(hyperparams['device'])
v_data_tensor = torch.tensor(v_data, dtype=torch.float32).to(hyperparams['device'])

pinn.X_collocation = X_collocation_tensor
pinn.X_ic = X_ic_tensor
pinn.expected_u0 = v_ic_tensor
pinn.X_boundary = X_boundary_tensor
pinn.normal_vectors = normal_vectors_tensor
pinn.X_data = X_data_tensor
pinn.expected_data = v_data_tensor

optimizer = optim.Adam(pinn.parameters(), lr=hyperparams['learning_rate'])

epochs = hyperparams['epochs']
batch_size = hyperparams['batch_size']
M_estimates = []
best_val_loss = float('inf')
no_improve_counter = 0
patience = 1000
best_model_path = os.path.join(results_dir, 'best_model.pth')

for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, data_loss_val, ode_loss, total_loss = pinn.train_step(optimizer, batch_size)
    if epoch % 100 == 0:
        M_current = pinn.M.item()
        M_estimates.append(M_current)
        print(f"Epoch {epoch}, Total Loss: {total_loss:.4e}, PDE: {pde_loss:.4e}, IC: {IC_loss:.4e}, BC: {BC_loss:.4e}, Data: {data_loss_val:.4e}, M: {M_current:.4e}")
        
        # Without separate validation data, use total_loss as proxy
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            no_improve_counter = 0
            pinn.save_model(best_model_path)
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print("Early stopping triggered.")
                break

# Save M predictions
M_predictions_file = os.path.join(results_dir, 'M_predictions.txt')
np.savetxt(M_predictions_file, M_estimates, fmt='%.6e')
print(f"M predictions saved to {M_predictions_file}")

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

hyperparams_file = os.path.join(results_dir, 'hyperparams.json')
with open(hyperparams_file, 'w') as f:
    json.dump(hyperparams, f, indent=4)
print(f"Hyperparameters saved to {hyperparams_file}")

pinn.load_model(best_model_path)
pinn.eval()

# Comparison times
comparison_times = [0.0, 0.1, 0.2, 0.4, 0.8, 0.99]

def plot_comparisons(model, solutions_fem, comparison_times, results_dir, device='cpu'):
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
            X_pinns_eval_tensor = torch.tensor(X_pinns_eval, dtype=torch.float32, device=device)
            y_pinns_pred = model.evaluate(X_pinns_eval_tensor)
            y_pinns_pred_np = y_pinns_pred.cpu().numpy().reshape(-1)

            error = np.abs(y_pinns_pred_np - fem_v)
            mse = mean_squared_error(fem_v, y_pinns_pred_np)
            mae = mean_absolute_error(fem_v, y_pinns_pred_np)
            rmse = np.sqrt(mse)

            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Comparisons at t = {t}', fontsize=16)

            cs1 = axs[0, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
            fig.colorbar(cs1, ax=axs[0, 0]).set_label('FEM Prediction')
            axs[0, 0].set_title('FEM Prediction')

            cs2 = axs[0, 1].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
            fig.colorbar(cs2, ax=axs[0, 1]).set_label('PINNs Prediction')
            axs[0, 1].set_title('PINNs Prediction')

            cs3 = axs[1, 0].tricontourf(triang, error, levels=50, cmap='viridis')
            fig.colorbar(cs3, ax=axs[1, 0]).set_label('Absolute Error |PINNs - FEM|')
            axs[1, 0].set_title('Absolute Error')

            # Re-plot FEM or optionally something else
            cs4 = axs[1, 1].tricontourf(triang, fem_v, levels=50, cmap='viridis')
            fig.colorbar(cs4, ax=axs[1, 1]).set_label('FEM again')
            axs[1, 1].set_title('FEM again')

            for ax_row in axs:
                for ax in ax_row:
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename = f'comparison_t_{t}.png'
            plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
            plt.close()
            print(f"Plots saved for time t = {t}")
            print(f"Time {t}: PINNs - MSE: {mse:.4e}, MAE: {mae:.4e}, RMSE: {rmse:.4e}")
            print(f"Plots saved as {plot_filename}\n")

plot_comparisons(pinn, solutions_fem_dict, comparison_times, results_dir, device=hyperparams['device'])

print("Comparison between FEM and PINNs simulations complete.")
