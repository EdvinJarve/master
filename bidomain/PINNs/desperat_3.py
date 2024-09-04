import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PINN(nn.Module):
    def __init__(self, num_inputs, num_layers, num_neurons, device):
        super(PINN, self).__init__()
        self.device = device
        activation = nn.SiLU()

        layers = [nn.Linear(num_inputs, num_neurons), activation]  # Input layer
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), activation]  # Hidden layers
        layers += [nn.Linear(num_neurons, 2)]  # Output layer for v and ue

        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)

def source_term(x, y, t, start_time=0.05, end_time=0.2, current_value=50):
    spatial_mask = (x <= 0.2) & (y >= 0.8)
    temporal_mask = (t >= start_time) & (t <= end_time)
    source = torch.where((spatial_mask) & (temporal_mask), 
                         torch.tensor(current_value, device=x.device, dtype=x.dtype), 
                         torch.tensor(0.0, device=x.device, dtype=x.dtype))
    return source

def pde(X_collocation, model, Mi):
    x = X_collocation[:, 0:1]  # X spatial coordinates
    y = X_collocation[:, 1:2]  # Y spatial coordinates
    t = X_collocation[:, 2:3]  # Time coordinates

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    outputs = model(torch.cat([x, y, t], dim=1))
    v = outputs[:, 0:1]
    ue = outputs[:, 1:2]

    # First derivatives
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    ue_x = torch.autograd.grad(ue, x, grad_outputs=torch.ones_like(ue), create_graph=True, retain_graph=True)[0]
    ue_y = torch.autograd.grad(ue, y, grad_outputs=torch.ones_like(ue), create_graph=True, retain_graph=True)[0]

    # Second derivatives
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    ue_xx = torch.autograd.grad(ue_x, x, grad_outputs=torch.ones_like(ue_x), create_graph=True)[0]
    ue_yy = torch.autograd.grad(ue_y, y, grad_outputs=torch.ones_like(ue_y), create_graph=True)[0]

    # Ensure Mi is a tensor
    if isinstance(Mi, np.ndarray):
        Mi = torch.tensor(Mi, dtype=torch.float32, device=x.device)

    # Calculate the Laplacian considering the tensor Mi
    laplacian_v = Mi[0, 0] * v_xx + Mi[1, 1] * v_yy
    laplacian_ue = Mi[0, 0] * ue_xx + Mi[1, 1] * ue_yy

    # Source term
    I_ion = source_term(x, y, t)

    # Residuals for the bidomain equations
    residual_1 = v_t - (laplacian_v + laplacian_ue) - I_ion
    residual_2 = Mi[0, 0] * v_xx + (Mi[0, 0] + Mi[1, 1]) * ue_yy

    # Combined residual
    residual = residual_1.pow(2).mean() + residual_2.pow(2).mean()

    return residual

def IC(x, model):
    x_space = x[:, 0:1]  # X spatial coordinates
    y_space = x[:, 1:2]  # Y spatial coordinates
    t_time = x[:, 2:3]  # Time coordinate

    # Create a tensor of zeros with the same shape as t_time to represent initial condition
    expected_u0 = torch.zeros_like(t_time)
    # Evaluate the model at the initial time
    u0 = model(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))

    # Calculate the squared error between the predicted and expected initial condition
    return (u0[:, 0:1] - expected_u0).pow(2).mean() + (u0[:, 1:2] - expected_u0).pow(2).mean()

def BC_neumann(x, model, normal_vectors):
    x.requires_grad_(True)
    outputs = model(x)
    v = outputs[:, 0:1]
    ue = outputs[:, 1:2]

    # Compute gradients of v and ue with respect to spatial coordinates
    gradients_v = torch.autograd.grad(outputs=v, inputs=x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    gradients_ue = torch.autograd.grad(outputs=ue, inputs=x, grad_outputs=torch.ones_like(ue), create_graph=True)[0]
    normal_flux_v = torch.sum(gradients_v * normal_vectors, dim=1)
    normal_flux_ue = torch.sum(gradients_ue * normal_vectors, dim=1)

    expected_value_v = torch.zeros_like(normal_flux_v)
    expected_value_ue = torch.zeros_like(normal_flux_ue)
    return (normal_flux_v - expected_value_v).pow(2).mean() + (normal_flux_ue - expected_value_ue).pow(2).mean()

def train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors, Mi):
    optimizer.zero_grad()

    IC_loss = (IC(X_ic, model))
    pde_loss = pde(X_collocation, model, Mi)
    BC_loss = BC_neumann(X_boundary, model, normal_vectors)

    total_loss = IC_loss + BC_loss + pde_loss
    total_loss.backward()
    optimizer.step()

    return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()

# Define number of points in the spatial and temporal domains
N_space = 30
N_time = 30

# Define the spatial and temporal coordinates of the training data
x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)  # additional y-space grid
x_time = np.linspace(0, 1, N_time)

# Remove the boundary points from the collocation points (used for evaluating the PDE)
x_collocation = x_space[1:-1]
y_collocation = y_space[1:-1]
t_collocation = x_time[1:-1]

# Create a meshgrid for collocation points in 2D space and time
x_collocation, y_collocation, t_collocation = np.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij')
x_collocation = x_collocation.flatten()
y_collocation = y_collocation.flatten()
t_collocation = t_collocation.flatten()

# Combine into a single array
X_collocation = np.vstack((x_collocation, y_collocation, t_collocation)).T

# Convert the coordinates to tensors on the chosen device
device = 'cpu'
print(f"Using device: {device}")
X_collocation = torch.tensor(X_collocation, dtype=torch.float32, device=device)

# Define the initial condition coordinates (spatial grid at t=0)
t_ic = np.zeros((N_space, N_space))
x_ic, y_ic = np.meshgrid(x_space, y_space, indexing='ij')
x_ic = x_ic.flatten()
y_ic = y_ic.flatten()
t_ic = t_ic.flatten()

# Combine the spatial and temporal coordinates
X_ic = np.vstack((x_ic, y_ic, t_ic)).T
X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)

# Define the boundary coordinates (all sides of the square at all time steps)
x_boundary = np.array([0, 1])
t_boundary = x_time
x_boundary, y_boundary, t_boundary = np.meshgrid(x_boundary, y_space, t_boundary, indexing='ij')
x_boundary = x_boundary.flatten()
y_boundary = y_boundary.flatten()
t_boundary = t_boundary.flatten()

# Repeat for the y-boundaries
y_boundary_side = np.array([0, 1])
x_side, y_side, t_side = np.meshgrid(x_space, y_boundary_side, x_time, indexing='ij')
x_side = x_side.flatten()
y_side = y_side.flatten()
t_side = t_side.flatten()

# Combine all boundary points
x_boundary_all = np.concatenate([x_boundary, x_side])
y_boundary_all = np.concatenate([y_boundary, y_side])
t_boundary_all = np.concatenate([t_boundary, t_side])

# Combine into a single array
X_boundary = np.vstack((x_boundary_all, y_boundary_all, t_boundary_all)).T
X_boundary = torch.tensor(X_boundary, dtype=torch.float32, device=device)

# Define boundary normal vectors
normal_vectors = np.zeros_like(X_boundary)
x_coords = X_boundary[:, 0]
y_coords = X_boundary[:, 1]

# Left boundary x=0
normal_vectors[(x_coords == 0), 0] = -1
# Right boundary x=1
normal_vectors[(x_coords == 1), 0] = 1
# Bottom boundary y=0
normal_vectors[(y_coords == 0), 1] = -1
# Top boundary y=1
normal_vectors[(y_coords == 1), 1] = 1

normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32, device=device)

# Ensure all tensors are on the same device
X_collocation = X_collocation.to(device)
X_ic = X_ic.to(device)
X_boundary = X_boundary.to(device)
normal_vectors = normal_vectors.to(device)

# Initialize the 2D model
model = PINN(num_inputs=3, num_layers=2, num_neurons=32, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 5000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Define the tensor Mi
theta = torch.tensor(torch.pi/4, dtype=torch.float32, device=device)
cos_theta = torch.cos(theta)
sin_theta = torch.sin(theta)
R = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=torch.float32, device=device)
D_i = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=device)
Mi = R.T @ D_i @ R

# Training loop
start_time = time.time()
for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, total_loss = train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors, Mi)
    
    if epoch % 100 == 0:
        loss_list.append(total_loss)
        epoch_list.append(epoch)
        scheduler.step()  # Update learning rate
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')

# Stop timer
end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

# Plot training loss over epochs
plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"figures_bi_pinns/loss_for_epochs={epochs}_corner_non_iso.pdf")
plt.show()

def get_predictions(model, x_grid, y_grid, t, device):
    # Flatten the grid points and create the input tensor
    x_flat = x_grid.flatten()[:, None]
    y_flat = y_grid.flatten()[:, None]
    t_flat = np.full_like(x_flat, t)
    
    input_tensor = torch.tensor(np.hstack((x_flat, y_flat, t_flat)), dtype=torch.float32, device=device)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(input_tensor)
    
    v_predictions = predictions[:, 0].cpu().numpy().reshape(x_grid.shape)
    ue_predictions = predictions[:, 1].cpu().numpy().reshape(x_grid.shape)
    
    return v_predictions, ue_predictions

# Define the spatial grid
x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_grid, y_grid = np.meshgrid(x_space, y_space)

# Define time points for evaluation
time_points = [0.0, 0.2, 0.4, 0.6, 1.0]

# Prepare for plotting
results_v = []
results_u_e = []
for time_point in time_points:
    v_pred, ue_pred = get_predictions(model, x_grid, y_grid, time_point, device)
    results_v.append(v_pred.flatten())
    results_u_e.append(ue_pred.flatten())

# Extract coordinates of degrees of freedom
x_coords = x_grid.flatten()
y_coords = y_grid.flatten()

# Plot solutions at different time points
for idx, time_point in enumerate(time_points):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))

    numerical_solution_v = results_v[idx]
    numerical_solution_u_e = results_u_e[idx]

    # Plot numerical solution for v
    ax = axes[0]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_v, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Numerical Solution for $v$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot numerical solution for u_e
    ax = axes[1]
    contour = ax.tricontourf(x_coords, y_coords, numerical_solution_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Numerical Solution for $u_e$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"figures_bi_pinns/bi_corner_non_iso_results_at_{time_point}.pdf")
    plt.show()

print("Visualization completed.")


"""
Computation time (excluding visualization): 7083.30 seconds

"""