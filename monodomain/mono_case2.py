import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
import ufl
import torch
from scipy.interpolate import griddata

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import MonodomainSolverPINNs

# Ensure directory for figures exists
os.makedirs('monodomain_results', exist_ok=True)

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
M = 1.0

def stimulus_expression(x, y, t, start_time=0.05, end_time=0.2, current_value=50):
    spatial_condition = ufl.And(x <= 0.2, y >= 0.8)
    temporal_condition = ufl.And(ufl.ge(t, start_time), ufl.le(t, end_time))
    return ufl.conditional(spatial_condition, ufl.conditional(temporal_condition, current_value, 0.0), 0.0)


# Define time points to store solutions
time_points = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0]

# Print statements for simulation steps
print("Solving second simulation with FEM")
start_time_fem = time.time()

# Initialize and run the FEM simulation
sim_fem = MonodomainSolverFEM(Nx, Ny, T, stimulus_expression, M, dt)
errors_fem, computation_time_fem, solutions_fem = sim_fem.run(time_points=time_points)
end_time_fem = time.time()
print(f"FEM simulation complete in {end_time_fem - start_time_fem:.2f} seconds")

# Extract coordinates of degrees of freedom
dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

def source_term(x, y, t, start_time=0.05, end_time=0.2, current_value=50):
    spatial_mask = (x <= 0.2) & (y >= 0.8)
    temporal_mask = (t >= start_time) & (t <= end_time)
    source = torch.where((spatial_mask) & (temporal_mask), 
                         torch.tensor(current_value, device=x.device, dtype=x.dtype), 
                         torch.tensor(0.0, device=x.device, dtype=x.dtype))
    return source


print("Solving second simulation with PINNs")
device = 'cpu'
print(f"Using device: {device}\n")
# Initialize the 2D model and prepare data
model = MonodomainSolverPINNs(num_inputs=3, num_layers=2, num_neurons=32, device=device, source_term_func=source_term, M = M)
model.prepare_data(Nx, Nt)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 7000 # Adjust as necessary 

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

plt.figure(figsize=(7, 5))
plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"monodomain_results/pinns_corner_loss_for_epochs={epochs}.pdf")
plt.close()


# Evaluate the PINNs solution at the given time points
solutions_pinns = {}
x_space = np.linspace(0, 1, Nx)
y_space = np.linspace(0, 1, Ny)
x_grid, y_grid = np.meshgrid(x_space, y_space)
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

for t in time_points:
    t_array = np.full_like(x_flat, fill_value=t)  # Same time for all spatial points
    model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
    model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(model_input_tensor).cpu().numpy()
    solutions_pinns[t] = predictions.flatten()  # Flatten the predictions

# Plot FEM and PINNs solutions side-by-side for comparison at each time point
for t in time_points:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    # Plot FEM numerical solution
    numerical_solution_fem = solutions_fem[t]
    if numerical_solution_fem is not None:
        numerical_solution_fem = numerical_solution_fem[:len(x_coords)]
        ax = axes[0]
        contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        ax.set_title(f'FEM Numerical Solution at t={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.set_title(f'No FEM solution at t={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(0, 1)  # Set y-axis range to be consistent
    
    # Plot PINNs numerical solution
    numerical_solution_pinns = solutions_pinns[t]
    ax = axes[1]
    contour = ax.tricontourf(x_flat, y_flat, numerical_solution_pinns, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'PINNs Numerical Solution at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(f"monodomain_results/mono_current_comparison_at_{t}.pdf")
    plt.close()

# Interpolation step before error calculation
numerical_solution_fem = solutions_fem[t]
numerical_solution_pinns = solutions_pinns[t]
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

for t in time_points:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate the absolute error between FEM and PINNs solutions
    numerical_solution_fem = solutions_fem[t]
    numerical_solution_pinns = solutions_pinns[t]
    if numerical_solution_fem is not None:
        numerical_solution_fem = numerical_solution_fem[:len(x_coords)]
        # Interpolate FEM solution to PINNs grid
        numerical_solution_fem_interp = griddata((x_coords, y_coords), numerical_solution_fem, (x_flat, y_flat), method='linear')
        error = np.abs(numerical_solution_fem_interp - numerical_solution_pinns)
        
        # Plot the error
        contour = ax.tricontourf(x_flat, y_flat, error, levels=50, cmap='hot')
        fig.colorbar(contour, ax=ax)
        ax.set_title(f'Error (|FEM - PINNs|) at t={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig(f"monodomain_results/error_comparison_at_{t}.pdf")
        plt.close()
    else:
        print(f"No FEM solution available at t={t}, skipping error plot.")
"""
Simulation complete in 0.05 seconds

Computation time (excluding visualization): 331.29 seconds
Epoch 5000, PDE Loss: 3.2331e-02, IC Loss: 4.3673e-04, BC Loss: 7.2011e-03, Total Loss: 3.9969e-02


"""