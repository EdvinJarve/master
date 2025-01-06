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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import qmc

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import MonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)

base_results_dir = os.path.join(project_root, 'monodomain_results', 'analytical_case')
os.makedirs(base_results_dir, exist_ok=True)

# =============================================================================
# Problem Setup
# =============================================================================

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T

Nx, Ny, Nt = 100, 100, 100
dt = T / Nt
M = 1.0

def analytical_solution_v(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

def source_term_func_pinns(x_spatial, t):
    pi = torch.pi
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    return (
        8 * pi**2 * torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.sin(t)
        + torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.cos(t)
    )

def source_term_func(x, y, t):
    return (
        8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.sin(t)
        + ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.cos(t)
    )

# =============================================================================
# FEM Simulation
# =============================================================================

domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
sim_fem = MonodomainSolverFEM(
    mesh=domain_mesh,
    T=T,
    dt=dt,
    M_i=M,
    source_term_func=source_term_func,
    initial_v=0.0
)

time_points = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
errors_fem, computation_time_fem, solutions_fem = sim_fem.run(time_points=time_points)

dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

triang = Triangulation(x_coords, y_coords)

# Save FEM data
all_times = np.array(time_points)
fem_matrix = np.vstack([solutions_fem[t] for t in time_points]).T
fem_data_file = os.path.join(base_results_dir, 'fem_data.npz')
np.savez(fem_data_file, x_coords=x_coords, y_coords=y_coords, time_points=all_times, fem_solutions=fem_matrix)
print(f"FEM data saved to {fem_data_file}.")

# =============================================================================
# PINNs Data Generation Using LHS
# =============================================================================

N_collocation = 20000
N_ic = 4000
N_bc = 4000
N_val = 2000
N_test = 2000
N_collocation_val = 1000
N_ic_val = 100
N_bc_val = 100

sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)
X_collocation = sample.copy()
X_collocation[:, 0] = x_min + (x_max - x_min) * sample[:, 0]
X_collocation[:, 1] = y_min + (y_max - y_min) * sample[:, 1]
X_collocation[:, 2] = t_min + (t_max - t_min) * sample[:, 2]

sampler_val_collocation = qmc.LatinHypercube(d=3)
sample_val_collocation = sampler_val_collocation.random(n=N_collocation_val)
X_collocation_val = sample_val_collocation.copy()
X_collocation_val[:, 0] = x_min + (x_max - x_min) * sample_val_collocation[:, 0]
X_collocation_val[:, 1] = y_min + (y_max - y_min) * sample_val_collocation[:, 1]
X_collocation_val[:, 2] = t_min + (t_max - t_min) * sample_val_collocation[:, 2]

# IC points
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)
X_ic = sample_ic.copy()
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]
X_ic = np.hstack((X_ic, np.zeros((N_ic, 1))))
expected_u0 = analytical_solution_v(X_ic[:, 0], X_ic[:, 1], X_ic[:, 2]).reshape(-1, 1)

sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)
X_ic_val = sample_ic_val.copy()
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]
X_ic_val = np.hstack((X_ic_val, np.zeros((N_ic_val, 1))))
expected_u0_val = analytical_solution_v(X_ic_val[:, 0], X_ic_val[:, 1], X_ic_val[:, 2]).reshape(-1, 1)

# BC points
N_per_boundary = N_bc // 4
def generate_boundary_points(N_per_boundary, x_fixed=None, y_fixed=None):
    sampler_boundary = qmc.LatinHypercube(d=2)
    sample_boundary = sampler_boundary.random(n=N_per_boundary)
    X_boundary = np.zeros((N_per_boundary, 3))
    if x_fixed is not None:
        X_boundary[:, 0] = x_fixed
        X_boundary[:, 1] = y_min + (y_max - y_min) * sample_boundary[:, 0]
    elif y_fixed is not None:
        X_boundary[:, 0] = x_min + (x_max - x_min) * sample_boundary[:, 0]
        X_boundary[:, 1] = y_fixed
    X_boundary[:, 2] = t_min + (t_max - t_min) * sample_boundary[:, 1]
    return X_boundary

X_left = generate_boundary_points(N_per_boundary, x_fixed=x_min)
X_right = generate_boundary_points(N_per_boundary, x_fixed=x_max)
X_bottom = generate_boundary_points(N_per_boundary, y_fixed=y_min)
X_top = generate_boundary_points(N_per_boundary, y_fixed=y_max)
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

normal_vectors_left = np.tile(np.array([[-1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_right = np.tile(np.array([[1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_bottom = np.tile(np.array([[0.0, -1.0]]), (N_per_boundary, 1))
normal_vectors_top = np.tile(np.array([[0.0, 1.0]]), (N_per_boundary, 1))
normal_vectors = np.vstack([normal_vectors_left, normal_vectors_right, normal_vectors_bottom, normal_vectors_top])

N_per_boundary_val = N_bc_val // 4
X_left_val = generate_boundary_points(N_per_boundary_val, x_fixed=x_min)
X_right_val = generate_boundary_points(N_per_boundary_val, x_fixed=x_max)
X_bottom_val = generate_boundary_points(N_per_boundary_val, y_fixed=y_min)
X_top_val = generate_boundary_points(N_per_boundary_val, y_fixed=y_max)
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

normal_vectors_left_val = np.tile(np.array([[-1.0, 0.0]]), (N_per_boundary_val, 1))
normal_vectors_right_val = np.tile(np.array([[1.0, 0.0]]), (N_per_boundary_val, 1))
normal_vectors_bottom_val = np.tile(np.array([[0.0, -1.0]]), (N_per_boundary_val, 1))
normal_vectors_top_val = np.tile(np.array([[0.0, 1.0]]), (N_per_boundary_val, 1))
normal_vectors_val = np.vstack([normal_vectors_left_val, normal_vectors_right_val, normal_vectors_bottom_val, normal_vectors_top_val])

device = 'cpu'
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

def scale_to_m1p1(X):
    return 2.0 * X - 1.0

# =============================================================================
# Function to run a single simulation for given architecture
# =============================================================================

def run_pinns_simulation(num_layers, num_neurons, scaling_func=None):
    run_dir = os.path.join(base_results_dir, f"results_layers={num_layers}_depth={num_neurons}")
    os.makedirs(run_dir, exist_ok=True)
    
    model_params = {
        'num_inputs': 3,
        'num_layers': num_layers,
        'num_neurons': num_neurons,
        'use_ode': False,
        'n_state_vars': 0,
        'loss_function': 'L2',
        'weight_strategy': 'dynamic',
        'alpha': 1.0,
        'domain_bounds': {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            't_min': t_min,
            't_max': t_max
        }
    }
    
    # We will add training_time and validation_time after we know their values
    params_file = os.path.join(run_dir, 'model_parameters.json')
    
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
        weight_strategy=model_params['weight_strategy'],
        alpha=model_params['alpha'],
        x_min=model_params['domain_bounds']['x_min'],
        x_max=model_params['domain_bounds']['x_max'],
        y_min=model_params['domain_bounds']['y_min'],
        y_max=model_params['domain_bounds']['y_max'],
        t_min=model_params['domain_bounds']['t_min'],
        t_max=model_params['domain_bounds']['t_max'],
        scaling_func=scaling_func
    )

    model.X_collocation = X_collocation_tensor
    model.X_ic = X_ic_tensor
    model.expected_u0 = expected_u0_tensor
    model.X_boundary = X_boundary_tensor
    model.normal_vectors = normal_vectors_tensor
    model.X_data = None
    model.expected_data = None

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 20000
    batch_size = 64
    patience = 1000
    best_val_loss = float('inf')
    no_improve_counter = 0
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    loss_list = []
    val_loss_list = []
    epoch_list = []

    start_time_pinns = time.time()
    for epoch in range(epochs + 1):
        pde_loss, IC_loss, BC_loss, data_loss_val, ode_loss, total_loss = model.train_step(optimizer, batch_size)
        
        model.eval()
        total_val_loss = model.validate(
            X_collocation_val=X_collocation_val_tensor,
            X_ic_val=X_ic_val_tensor,
            expected_u0_val=expected_u0_val_tensor,
            X_boundary_val=X_boundary_val_tensor,
            normal_vectors_val=normal_vectors_val_tensor
        )

        if epoch % 100 == 0:
            loss_list.append(total_loss)
            val_loss_list.append(total_val_loss)
            epoch_list.append(epoch)
            print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, '
                  f'BC Loss: {BC_loss:.4e}, ODE Loss: {ode_loss:.4e}, '
                  f'Total Loss: {total_loss:.4e}, Validation Loss: {total_val_loss:.4e}')

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                no_improve_counter = 0
                model.save_model(best_model_path)
                print(f"New best model saved with validation loss {best_val_loss:.4e}")
            else:
                no_improve_counter += 1
                if no_improve_counter >= patience:
                    print("Early stopping triggered.")
                    break

    end_time_pinns = time.time()
    computation_time_pinns = end_time_pinns - start_time_pinns

    model.load_model(best_model_path)

    # Compute training_time and add it to model_params
    model_params['training_time'] = computation_time_pinns

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, loss_list, label='Training Loss')
    plt.plot(epoch_list, val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'), dpi=300)
    plt.close()
    print("Loss plot saved.\n")

    # Measure validation_time (test evaluation)
    val_start = time.time()
    X_fem_test = np.column_stack((x_coords, y_coords, np.full_like(x_coords, T)))
    X_fem_test_tensor = torch.tensor(X_fem_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pinns_pred = model(X_fem_test_tensor).cpu().numpy().reshape(-1)
    val_end = time.time()
    validation_time = val_end - val_start

    # Compute test MSE and relative MSE
    fem_solution_T = solutions_fem[T]
    test_mse = mean_squared_error(fem_solution_T, y_pinns_pred)
    relative_mse = np.sum((y_pinns_pred - fem_solution_T)**2) / np.sum(fem_solution_T**2)

    # Add validation_time to model_params
    model_params['validation_time'] = validation_time

    # Save updated model_params to JSON
    with open(params_file, 'w') as f:
        json.dump(model_params, f, indent=4)

    # Comparison plots
    comparison_times = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    for tt in comparison_times:
        if tt not in solutions_fem:
            continue
        fem_v = solutions_fem[tt]
        X_pinns_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, tt)))
        X_pinns_eval_tensor = torch.tensor(X_pinns_eval, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pinns_pred_np = model(X_pinns_eval_tensor).cpu().numpy().reshape(-1)

        u_analytical = analytical_solution_v(x_coords, y_coords, tt)
        pinns_error = np.abs(y_pinns_pred_np - u_analytical)
        fem_error = np.abs(fem_v - u_analytical)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparisons at t = {tt}', fontsize=16)

        cs1 = axs[0, 0].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
        fig.colorbar(cs1, ax=axs[0, 0]).set_label('PINNs Prediction')
        axs[0, 0].set_title('PINNs Prediction')

        cs2 = axs[0, 1].tricontourf(triang, pinns_error, levels=50, cmap='viridis')
        fig.colorbar(cs2, ax=axs[0, 1]).set_label('PINNs Error |PINNs - Analytical|')
        axs[0, 1].set_title('PINNs Absolute Error')

        cs3 = axs[1, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
        fig.colorbar(cs3, ax=axs[1, 0]).set_label('FEM Solution')
        axs[1, 0].set_title('FEM Prediction')

        cs4 = axs[1, 1].tricontourf(triang, fem_error, levels=50, cmap='viridis')
        fig.colorbar(cs4, ax=axs[1, 1]).set_label('FEM Error |FEM - Analytical|')
        axs[1, 1].set_title('FEM Absolute Error')

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'comparison_t_{tt}.png'
        plt.savefig(os.path.join(run_dir, plot_filename), dpi=300)
        plt.close()
        print(f"Plots saved for time t = {tt}")

    return test_mse, relative_mse

# =============================================================================
# Main Loop Over Different Architectures
# =============================================================================

num_layers_list = [2]
num_neurons_list = [128, 256]

test_mse_dict = {}
test_relative_mse_dict = {}

scaling_func = scale_to_m1p1  # or None if you don't want scaling

for depth in num_neurons_list:
    for layers in num_layers_list:
        test_mse_val, test_relative_mse_val = run_pinns_simulation(layers, depth, scaling_func=scaling_func)
        test_mse_dict[(layers, depth)] = test_mse_val
        test_relative_mse_dict[(layers, depth)] = test_relative_mse_val

# Write test MSE to file
test_mse_file = os.path.join(base_results_dir, 'test_mse.txt')
with open(test_mse_file, 'w') as f:
    for depth in num_neurons_list:
        mse_values = [f"{test_mse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(mse_values) + "\n")

# Write test Relative MSE to file
test_relative_mse_file = os.path.join(base_results_dir, 'test_relative_mse.txt')
with open(test_relative_mse_file, 'w') as f:
    for depth in num_neurons_list:
        rel_mse_values = [f"{test_relative_mse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(rel_mse_values) + "\n")

print("All simulations complete. Test MSE and relative MSE values saved.")
