import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
import torch
import ufl
from matplotlib.ticker import ScalarFormatter


# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heart_solver_fem import MonodomainSolverFEM
from utils.heart_solver_pinns import MonodomainSolverPINNs

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure directory for figures exists
os.makedirs('monodomain_results', exist_ok=True)

# Define the analytical solution functions
def analytical_solution_v(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

# Define mesh parameters and temporal parameters
Nx, Ny, Nt = 20, 20, 20
T = 1.0
dt = T / Nt
M = 1.0

# Define source term Istim for FEM using UFL
def stimulus_expression(x, y, t):
    return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x) * ufl.cos(2 * ufl.pi * y) * ufl.sin(t)

# Print statements for simulation steps
print("Solving first simulation with FEM")
start_time_fem = time.time()

# Initialize and run the FEM simulation
sim_fem = MonodomainSolverFEM(Nx, Ny, T, stimulus_expression, M, dt)

mse_fem, computation_time_fem = sim_fem.run(analytical_solution_v=analytical_solution_v)

end_time_fem = time.time()
print(f"Simulation complete in {end_time_fem - start_time_fem:.2f} seconds\n")

# Extract coordinates of degrees of freedom
dof_coords = sim_fem.V.tabulate_dof_coordinates()
x_coords = dof_coords[:, 0]
y_coords = dof_coords[:, 1]

# Extract the data for plotting
numerical_solution_fem = sim_fem.v_h.x.array
analytical_solution_values_fem = analytical_solution_v(x_coords, y_coords, T)

# Define the analytical solution for PINNs
def analytical_solution(x):
    x_coord = x[:, 0:1]  # x spatial coordinates
    y_coord = x[:, 1:2]  # y spatial coordinates
    t_coord = x[:, 2:3]  # Time coordinates

    return np.cos(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord) * np.sin(t_coord)

# Define source term function for PINNs
def source_term_func(x, y, t):
    pi = torch.pi
    return 8 * pi**2 * torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.sin(t)

device = 'cpu'

# Initialize the 2D model and prepare data
model = MonodomainSolverPINNs(num_inputs=3, num_layers=2, num_neurons=32, device=device, source_term_func=source_term_func, M=M)
model.prepare_data(Nx, Nt)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 5000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Training loop
print("Solving first simulation with PINNs")
print(f"Using device: {device}\n")
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
plt.savefig(f"monodomain_results/pinns_analytical_loss_for_epochs={epochs}.pdf")
plt.close()

# Define the spatial grid
x_space = np.linspace(0, 1, Nx)
y_space = np.linspace(0, 1, Ny)
x_grid, y_grid = np.meshgrid(x_space, y_space)

# Flatten the grid for processing
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Initialize list to store errors
mse_pinns = []

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
    analytical_solution_values = np.cos(2 * np.pi * x_flat) * np.cos(2 * np.pi * y_flat) * np.sin(t_array)
    
    # Reshape data for plotting
    predictions = predictions.reshape(Nx, Ny)
    analytical_solution_values = analytical_solution_values.reshape(Nx, Ny)
    
    # Calculate and store error
    error = np.linalg.norm(predictions - analytical_solution_values) / len(predictions.flatten())
    mse_pinns.append(error)

# Plot numerical and analytical solutions at final time for both PINNs and FEM
fig, axes = plt.subplots(1, 2, figsize=(11, 6))

# Plot FEM numerical solution at T = 1.0
ax = axes[0]
contour = ax.tricontourf(x_coords, y_coords, numerical_solution_fem, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'FEM Numerical Solution at T={T:.1f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot PINNs numerical solution at T = 1.0
t_final = 1.0
t_array = np.full_like(x_flat, fill_value=t_final)
model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
with torch.no_grad():
    predictions_pinns = model(model_input_tensor).cpu().numpy()

predictions_pinns = predictions_pinns.reshape(Nx, Ny)

ax = axes[1]
contour = ax.contourf(X, Y, predictions_pinns, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax)
ax.set_title(f'PINNs Numerical Solution at T={t_final}')
ax.set_xlabel('x')
ax.set_ylabel('y')


plt.tight_layout()
plt.savefig(f"monodomain_results/analytical_comparison_at_T={t_final}.pdf")
plt.close()



# Plot the error over time for PINNs
plt.figure(figsize=(7, 5))
plt.plot(np.linspace(0, T, len(mse_fem)), mse_fem, label = "FEM")
plt.plot(np.linspace(0, 1, len(mse_pinns)), mse_pinns, label = "PINNs")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title("MSE over Time")
ax = plt.gca()  # Get current axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.savefig(f"monodomain_results/mono_analytical_mse_over_time.pdf")
plt.close()

"""
Simulation complete in 0.05 seconds

Epoch 5000, PDE Loss: 4.1673e-02, IC Loss: 3.7265e-04, BC Loss: 4.6467e-03, Total Loss: 4.6692e-02
Computation time (excluding visualization): 339.36 seconds



"""
