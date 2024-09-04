import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
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
        activation = nn.Tanh()

        layers = [nn.Linear(num_inputs, num_neurons), activation]  # Input layer
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), activation]  # Hidden layers
        layers += [nn.Linear(num_neurons, 1)]  # Output layer

        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)

def source_term(x, y, t):
    current_duration = 0.2
    current_value = 200  # Value of the applied current

    # Create a mask for the spatial region and time frame
    spatial_mask = (x <= 0.05) & (y >= 0.95)
    temporal_mask = (t <= current_duration)

    # Apply the current where both spatial and temporal conditions are met
    source = torch.where(spatial_mask & temporal_mask, current_value, torch.tensor(0.0, device=x.device))

    return source

def pde(X_collocation, model_v, model_u_e):
    x = X_collocation[:, 0:1]  # X spatial coordinates
    y = X_collocation[:, 1:2]  # Y spatial coordinates
    t = X_collocation[:, 2:3]  # Time coordinates

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    v = model_v(torch.cat([x, y, t], dim=1))
    u_e = model_u_e(torch.cat([x, y, t], dim=1))

    # First derivatives
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    u_e_t = torch.autograd.grad(u_e, t, grad_outputs=torch.ones_like(u_e), create_graph=True, retain_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    u_e_x = torch.autograd.grad(u_e, x, grad_outputs=torch.ones_like(u_e), create_graph=True, retain_graph=True)[0]
    u_e_y = torch.autograd.grad(u_e, y, grad_outputs=torch.ones_like(u_e), create_graph=True, retain_graph=True)[0]

    # Second derivatives
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    u_e_xx = torch.autograd.grad(u_e_x, x, grad_outputs=torch.ones_like(u_e_x), create_graph=True)[0]
    u_e_yy = torch.autograd.grad(u_e_y, y, grad_outputs=torch.ones_like(u_e_y), create_graph=True)[0]

    # Incorporate isotropic conductivities (assuming D_i = D_e = identity matrices)
    laplacian_v = v_xx + v_yy
    laplacian_u_e = u_e_xx + u_e_yy

    # Source term from applied current
    source = source_term(x, y, t)

    # Residuals for bidomain model equations
    residual_v = v_t - laplacian_v - source  # Intracellular potential equation
    residual_u_e = u_e_t - laplacian_u_e + laplacian_v  # Extracellular potential equation

    return residual_v.pow(2).mean() + residual_u_e.pow(2).mean()



def IC(x, model_v, model_u_e):
    x_space = x[:, 0:1]  # X spatial coordinates
    y_space = x[:, 1:2]  # Y spatial coordinates
    t_time = x[:, 2:3]  # Time coordinate

    # Create a tensor of zeros with the same shape as t_time to represent initial condition
    expected_u0 = torch.zeros_like(t_time)
    # Evaluate the model at the initial time
    v0 = model_v(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))
    u_e0 = model_u_e(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))

    # Calculate the squared error between the predicted and expected initial condition
    return (v0 - expected_u0).pow(2).mean() + (u_e0 - expected_u0).pow(2).mean()



def BC_neumann(x, model_v, model_u_e, normal_vectors):
    x.requires_grad_(True)
    v = model_v(x)
    u_e = model_u_e(x)

    # Compute gradients of v and u_e with respect to spatial coordinates
    gradients_v = torch.autograd.grad(outputs=v, inputs=x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    gradients_u_e = torch.autograd.grad(outputs=u_e, inputs=x, grad_outputs=torch.ones_like(u_e), create_graph=True)[0]
    
    normal_flux_v = torch.sum(gradients_v * normal_vectors, dim=1)
    normal_flux_u_e = torch.sum(gradients_u_e * normal_vectors, dim=1)

    expected_value = torch.zeros_like(normal_flux_v)
    return (normal_flux_v - expected_value).pow(2).mean() + (normal_flux_u_e - expected_value).pow(2).mean()

def train_step(X_boundary, X_collocation, X_ic, optimizer, model_v, model_u_e, normal_vectors):
    optimizer.zero_grad()

    IC_loss = IC(X_ic, model_v, model_u_e)
    pde_loss = pde(X_collocation, model_v, model_u_e)
    BC_loss = BC_neumann(X_boundary, model_v, model_u_e, normal_vectors)

    total_loss = IC_loss + BC_loss + pde_loss
    total_loss.backward()
    optimizer.step()

    return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()


# Define number of points in the spatial and temporal domains
N_space = 20
N_time = 20

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# Initialize the 2D models for v and u_e
model_v = PINN(num_inputs=3, num_layers=4, num_neurons=32, device=device)
model_u_e = PINN(num_inputs=3, num_layers=4, num_neurons=32, device=device)
optimizer = torch.optim.Adam(list(model_v.parameters()) + list(model_u_e.parameters()), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 2000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Training loop
start_time = time.time()
for epoch in range(epochs + 1):
    pde_loss, IC_loss, BC_loss, total_loss = train_step(X_boundary, X_collocation, X_ic, optimizer, model_v, model_u_e, normal_vectors)
    
    if epoch % 100 == 0:
        loss_list.append(total_loss)
        epoch_list.append(epoch)
        scheduler.step()  # Update learning rate
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')

# Stop timer
end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"figures/loss_for_epochs={epochs}_corner.pdf")

# Define the spatial grid
N_space = 20  
x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_grid, y_grid = np.meshgrid(x_space, y_space)

# Flatten the grid for processing
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Define time points for evaluation
time_points = [0.0, 0.2, 0.4, 0.6, 1.0]

# Prepare for plotting
fig, axes = plt.subplots(nrows=len(time_points), ncols=2, figsize=(20, 20))

# Evaluate at each time point
for idx, t in enumerate(time_points):
    # Create input tensor for the model
    t_array = np.full_like(x_flat, fill_value=t)  # Same time for all spatial points
    model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
    model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
    
    # Predict with the models
    model_v.eval()
    model_u_e.eval()
    with torch.no_grad():
        predictions_v = model_v(model_input_tensor).cpu().numpy().flatten()
        predictions_u_e = model_u_e(model_input_tensor).cpu().numpy().flatten()

    # Plot numerical solution for v
    ax = axes[idx, 0]
    contour_v = ax.tricontourf(x_flat, y_flat, predictions_v, levels=50, cmap='viridis')
    fig.colorbar(contour_v, ax=ax)
    ax.set_title(f'Numerical Solution for $v$ at $t={t}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot numerical solution for u_e
    ax = axes[idx, 1]
    contour_u_e = ax.tricontourf(x_flat, y_flat, predictions_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour_u_e, ax=ax)
    ax.set_title(f'Numerical Solution for $u_e$ at $t={t}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

fig.suptitle(f"PINNs solution of bidomain model for $x, y \\in [0,1]$ at different time points", fontweight='bold')
plt.tight_layout()
plt.savefig(f"figures/Comparison_PINNs_corner_current.pdf")
plt.show()