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
from matplotlib.ticker import ScalarFormatter

device = 'cpu'

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


def stimulus_expression(x, y, t):
    return ufl.cos(t) * ufl.cos(2*ufl.pi * x) * ufl.cos(2 * ufl.pi * y) + 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.sin(t)

# Define source term function for PINNs
def source_term_func(x, y, t):
    pi = torch.pi
    return torch.cos(t) * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)\
         + 4 * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y) * torch.sin(t)

# Define the analytical solution functions
def analytical_solution_v(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t) 

def analytical_solution_u_e(x, y, t):
    return -np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t) / 2.0

# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 20, 20, 20
T = 1.0
dt = T / Nt
Mi, Me = 1.0, 1.0

print("Solving simulation with FEM")
start_time_fem = time.time()

# Initialize and run the FEM simulation
sim_fem = BidomainSolverFEM(Nx, Ny, T, stimulus_expression, Mi, Me, dt)
errors_fem_v, errors_fem_u_e, computation_time_fem = sim_fem.run(analytical_solution_v=analytical_solution_v, analytical_solution_u_e= analytical_solution_u_e)

end_time_fem = time.time()
print(f"Simulation complete in {end_time_fem - start_time_fem:.2f} seconds\n")

# Extract coordinates of degrees of freedom
dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

# Extract the data for plotting
numerical_solution_fem_v = sim_fem.v_h.x.array
numerical_solution_fem_u_e = sim_fem.u_e_h.x.array
analytical_solution_values_fem_v = analytical_solution_v(x_coords, y_coords, T)

# Initialize the 2D model and prepare data
model = BidomainSolverPINNs(num_inputs=3, num_layers=2, num_neurons=32, device=device, source_term_func=source_term_func, Mi=Mi, Me=Me, )
model.prepare_data(Nx, Nt)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 5000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Training loop
print("Solving simulation with PINNs")
print(f"Using device: {device}\n")
start_time_pinns = time.time()
for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, total_loss = model.train_step(optimizer)
    
    if epoch % 100 == 0:  # Adjust the interval as needed
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
plt.savefig(f"bidomain_results/pinns_analytical_loss_for_epochs={epochs}.pdf")
plt.show()

# Define the spatial grid
x_space = np.linspace(0, 1, Nx)
y_space = np.linspace(0, 1, Ny)
x_grid, y_grid = np.meshgrid(x_space, y_space)

# Flatten the grid for processing
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Initialize list to store errors
errors_pinns_v = []
errors_pinns_u_e = []

# Create a meshgrid for contour plotting
X, Y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))

# Evaluate at each time point
for t in np.linspace(0, 1, Nt):
    # Create input tensor for the model
    t_array = np.full_like(x_flat, fill_value=t)  # Same time for all spatial points
    model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
    model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
    
    # Predict with the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(model_input_tensor).cpu().numpy()

    # Compute the analytical solution
    analytical_solution_values_v = analytical_solution_v(x_flat, y_flat, t)
    analytical_solution_values_u_e = analytical_solution_u_e(x_flat, y_flat, t)
    
    # Reshape data for plotting
    predictions_v = predictions[:, 0].reshape(Nx, Ny)
    predictions_u_e = predictions[:, 1].reshape(Nx, Ny)
    
    # Calculate and store error
    error_v = np.linalg.norm(predictions_v - analytical_solution_values_v.reshape(Nx, Ny)) / len(predictions_v.flatten())
    errors_pinns_v.append(error_v)
    
    error_u_e = np.linalg.norm(predictions_u_e - analytical_solution_values_u_e.reshape(Nx, Ny)) / len(predictions_u_e.flatten())
    errors_pinns_u_e.append(error_u_e)

# Plot numerical and analytical solutions at final time for both PINNs and FEM
fig, axes = plt.subplots(1, 2, figsize=(11, 6))

# Plot FEM numerical solution at T = 1.0 for v
ax = axes[0]
contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem_v, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'FEM Numerical Solution for v at T={T:.1f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot PINNs numerical solution at T = 1.0 for v
t_final = 1.0
t_array = np.full_like(x_flat, fill_value=t_final)
model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
with torch.no_grad():
    predictions_pinns = model(model_input_tensor).cpu().numpy()

predictions_pinns_v = predictions_pinns[:, 0].reshape(Nx, Ny)

ax = axes[1]
contour = ax.contourf(X, Y, predictions_pinns_v, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'PINNs Numerical Solution for v at T={t_final}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(f"bidomain_results/analytical_comparison_v_at_T={t_final}.pdf")

# Plot numerical and analytical solutions at final time for both PINNs and FEM for u_e
fig, axes = plt.subplots(1, 2, figsize=(11, 6))

# Plot FEM numerical solution at T = 1.0 for u_e
ax = axes[0]
contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem_u_e, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'FEM Numerical Solution for u_e at T={T:.1f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot PINNs numerical solution at T = 1.0 for u_e
predictions_pinns_u_e = predictions_pinns[:, 1].reshape(Nx, Ny)

ax = axes[1]
contour = ax.contourf(X, Y, predictions_pinns_u_e, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'PINNs Numerical Solution for u_e at T={t_final}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(f"bidomain_results/analytical_comparison_u_e_at_T={t_final}.pdf")
plt.close()

# Plot the MSE over time for FEM
plt.figure(figsize=(7, 5))
plt.plot(np.linspace(0, T, len(errors_fem_v)), errors_fem_v, label="FEM $v$")
plt.plot(np.linspace(0, 1, len(errors_pinns_v)), errors_pinns_v, label="PINNs $v$")
plt.plot(np.linspace(0, T, len(errors_fem_u_e)), errors_fem_u_e, label="FEM $u_e$")
plt.plot(np.linspace(0, 1, len(errors_pinns_u_e)), errors_pinns_u_e, label="PINNs $u_e$")
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("MSE over time Bidomain")
ax = plt.gca()  # Get current axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.savefig(f"bidomain_results/bi_ue_mse_over_time.pdf")
plt.close()