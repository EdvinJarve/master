import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
import torch
import ufl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.heart_solver_fem import BidomainSolverFEM
from utils.heart_solver_pinns import BidomainSolverPINNs
from scipy.interpolate import griddata

device = 'cpu'

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure directory for figures exists
os.makedirs('bidomain_results', exist_ok=True)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 30, 30, 30
T = 1.0
dt = T / Nt
M_e = M_i = ufl.as_matrix([[1, -1], [-1, 1]])

def stimulus_expression(x, y, t, start_time=0.05, end_time=0.2, current_value=50):
    spatial_condition = ufl.And(x <= 0.2, y >= 0.8)
    temporal_condition = ufl.And(ufl.ge(t, start_time), ufl.le(t, end_time))
    return ufl.conditional(spatial_condition, ufl.conditional(temporal_condition, current_value, 0.0), 0.0)

def source_term(x, y, t, start_time=0.05, end_time=0.2, current_value=50):
    spatial_mask = (x <= 0.2) & (y >= 0.8)
    temporal_mask = (t >= start_time) & (t <= end_time)
    source = torch.where((spatial_mask) & (temporal_mask), 
                         torch.tensor(current_value, device=x.device, dtype=x.dtype), 
                         torch.tensor(0.0, device=x.device, dtype=x.dtype))
    return source

# Define time points to store solutions
time_points = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0]

# Print statements for simulation steps
print("Solving simulation with FEM")
start_time_fem = time.time()

# Initialize and run the FEM simulation
sim_fem = BidomainSolverFEM(Nx, Ny, T, stimulus_expression, M_i, M_e, dt)
errors_fem_v, errors_fem_u_e, computation_time_fem, solutions_fem_v, solutions_fem_u_e = sim_fem.run(time_points=time_points)

end_time_fem = time.time()
print(f"FEM simulation complete in {end_time_fem - start_time_fem:.2f} seconds")

# Extract coordinates of degrees of freedom
dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]
M_i = M_e = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)

print("Solving simulation with PINNs")
device = 'cpu'
print(f"Using device: {device}\n")
# Initialize the 2D model and prepare data
model = BidomainSolverPINNs(num_inputs=3, num_layers=2, num_neurons=32, device=device, source_term_func=source_term, Mi=M_i, Me=M_e)
model.prepare_data(Nx, Nt)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 7000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Training loop
start_time_pinns = time.time()
for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, total_loss = model.train_step(optimizer)
    
    if epoch % 100 == 0:
        loss_list.append(total_loss)
        epoch_list.append(epoch)
        scheduler.step()  # Update learning rate
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')

end_time_pinns = time.time()
computation_time_pinns = end_time_pinns - start_time_pinns
print(f"Computation time (excluding visualization): {computation_time_pinns:.2f} seconds")

plt.figure(figsize=(8, 5))
plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"bidomain_results/pinns_noniso_corner_loss_for_epochs={epochs}.pdf")
plt.close()

# Evaluate the PINNs solution at the given time points
solutions_pinns_v = {}
solutions_pinns_u_e = {}
x_space = np.linspace(0, 1, Nx)
y_space = np.linspace(0, 1, Ny)
x_grid, y_grid = np.meshgrid(x_space, y_space)
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

solutions_pinns_v = {}
solutions_pinns_u_e = {}

for t in time_points:
    t_array = np.full_like(x_flat, fill_value=t)  # Same time for all spatial points
    model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
    model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(model_input_tensor).cpu().numpy()
    solutions_pinns_v[t] = predictions[:, 0].flatten()  # Flatten the predictions for v
    solutions_pinns_u_e[t] = predictions[:, 1].flatten()  # Flatten the predictions for u_e

# Plot FEM and PINNs solutions side-by-side for comparison at each time point
for t in time_points:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot FEM numerical solution for v
    numerical_solution_fem_v = solutions_fem_v[t]
    ax = axes[0]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem_v, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'FEM Numerical Solution for v at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot PINNs numerical solution for v
    numerical_solution_pinns_v = solutions_pinns_v[t]
    ax = axes[1]
    contour = ax.tricontourf(x_flat, y_flat, numerical_solution_pinns_v, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'PINNs Numerical Solution for v at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(f"bidomain_results/non_iso_v_comparison_at_t_{t}.pdf")
    plt.close()

for t in time_points:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    # Plot FEM numerical solution for u_e
    numerical_solution_fem_u_e = solutions_fem_u_e[t]
    ax = axes[0]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'FEM Numerical Solution for u_e at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot PINNs numerical solution for u_e
    numerical_solution_pinns_u_e = solutions_pinns_u_e[t]
    ax = axes[1]
    contour = ax.tricontourf(x_flat, y_flat, numerical_solution_pinns_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'PINNs Numerical Solution for u_e at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"bidomain_results/non_iso_u_e_comparison_at_t_{t}.pdf")
    plt.close()


for t in time_points:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate the absolute error between FEM and PINNs solutions for v
    numerical_solution_fem_v = solutions_fem_v[t]
    numerical_solution_pinns_v = solutions_pinns_v[t]
    if numerical_solution_fem_v is not None:
        numerical_solution_fem_v = numerical_solution_fem_v[:len(x_coords)]
        # Interpolate FEM solution to PINNs grid
        numerical_solution_fem_v_interp = griddata((x_coords, y_coords), numerical_solution_fem_v, (x_flat, y_flat), method='linear')
        error_v = np.abs(numerical_solution_fem_v_interp - numerical_solution_pinns_v)
        
        # Plot the error for v
        contour_v = ax.tricontourf(x_flat, y_flat, error_v, levels=50, cmap='hot')
        fig.colorbar(contour_v, ax=ax)
        ax.set_title(f'Error (|FEM - PINNs|) for v at t={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig(f"bidomain_results/error_v_comparison_at_{t}.pdf")
        plt.close()
    else:
        print(f"No FEM solution available at t={t}, skipping error plot for v.")

for t in time_points:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate the absolute error between FEM and PINNs solutions for u_e
    numerical_solution_fem_u_e = solutions_fem_u_e[t]
    numerical_solution_pinns_u_e = solutions_pinns_u_e[t]
    if numerical_solution_fem_u_e is not None:
        numerical_solution_fem_u_e = numerical_solution_fem_u_e[:len(x_coords)]
        # Interpolate FEM solution to PINNs grid
        numerical_solution_fem_u_e_interp = griddata((x_coords, y_coords), numerical_solution_fem_u_e, (x_flat, y_flat), method='linear')
        error_u_e = np.abs(numerical_solution_fem_u_e_interp - numerical_solution_pinns_u_e)
        
        # Plot the error for u_e
        contour_u_e = ax.tricontourf(x_flat, y_flat, error_u_e, levels=50, cmap='hot')
        fig.colorbar(contour_u_e, ax=ax)
        ax.set_title(f'Error (|FEM - PINNs|) for u_e at t={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig(f"bidomain_results/error_u_e_comparison_at_{t}.pdf")
        plt.close()
    else:
        print(f"No FEM solution available at t={t}, skipping error plot for u_e.")

