import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

"""
This code uses PINNs to solve the monodomain model in 2D meshgrid. Spesifically, the code seeks to reproduce the results from
https://finsberg.github.io/fenics-beat/tests/README.html to check the credability of the solver. The equation we solve 
reduces to

dv/dt = ∇²v + I_app which is essentially a diffusion equation with a source term.

The code is based on code from Morten Hjort-Jensen with modifications (https://github.com/Gocoderunav/Diffusion_Equation-_PINNS)

"""

class PINN(nn.Module):
    def __init__(self, num_inputs, num_layers, num_neurons, device):
        super(PINN, self).__init__()
        self.device = device
        activation = nn.SiLU()

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
    

def pde(X_collocation, model):
    x = X_collocation[:, 0:1]  # X spatial coordinates
    y = X_collocation[:, 1:2]  # Y spatial coordinates
    t = X_collocation[:, 2:3]  # Time coordinates

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    u = model(torch.cat([x, y, t], dim=1))

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # Source term
    pi = torch.pi
    source_term = 8 * pi**2 * torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.sin(t)

    # Laplacian(u) - u_t should equal -source_term according to the heat/diffusion equation with a source
    laplacian_u = u_xx + u_yy
    residual = u_t - laplacian_u - source_term  

    return residual.pow(2)


def IC(x, model):
    x_space = x[:, 0:1]  # X spatial coordinates
    y_space = x[:, 1:2]  # Y spatial coordinates
    t_time = x[:, 2:3]  # Time coordinate

    # Create a tensor of zeros with the same shape as t_time to represent initial condition
    expected_u0 = torch.zeros_like(t_time)
    # Evaluate the model at the initial time
    u0 = model(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))

    # Calculate the squared error between the predicted and expected initial condition
    return (u0 - expected_u0).pow(2).mean()


def BC_neumann(x, model, normal_vectors):
    x.requires_grad_(True)
    u = model(x)

    # Compute gradients of u with respect to spatial coordinates
    gradients = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    normal_flux = torch.sum(gradients * normal_vectors, dim=1)

    expected_value = torch.zeros_like(normal_flux)
    return (normal_flux - expected_value).pow(2).mean()


def train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors):
    optimizer.zero_grad()

    IC_loss = torch.mean(IC(X_ic, model))
    pde_loss = torch.mean(pde(X_collocation, model))
    BC_loss = BC_neumann(X_boundary, model, normal_vectors)

    total_loss = IC_loss + BC_loss + pde_loss
    total_loss.backward()
    optimizer.step()

    return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()


def analytical_solution(x):
    x_coord = x[:, 0:1]  # x spatial coordinates
    y_coord = x[:, 1:2]  # y spatial coordinates
    t_coord = x[:, 2:3]  # Time coordinates

    return np.cos(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord) * np.sin(t_coord)


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

# Initialize the 2D model
model = PINN(num_inputs=3, num_layers=2, num_neurons=32, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 10000  # Adjust as necessary 

# Lists to store loss data for plotting
loss_list = []
epoch_list = []

# Training loop
for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, total_loss = train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors)
    
    if epoch % 100 == 0:
        loss_list.append(total_loss)
        epoch_list.append(epoch)
        scheduler.step()  # Update learning rate
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')

plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"figures/loss_for_epochs={epochs}.pdf")

# Define the spatial grid
N_space = 20  
x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_grid, y_grid = np.meshgrid(x_space, y_space)

# Flatten the grid for processing
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

# Define time points for evaluation
time_points = [0.5, 1] 

# Prepare for plotting
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Evaluate at each time point
for idx, t in enumerate(time_points):
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
    predictions = predictions.reshape(N_space, N_space)
    analytical_solution_values = analytical_solution_values.reshape(N_space, N_space)

    # Plot numerical solution
    ax = axes[3*idx]
    contour = ax.contourf(x_grid, y_grid, predictions, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Numerical Solution at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot analytical solution
    ax = axes[3*idx + 1]
    contour = ax.contourf(x_grid, y_grid, analytical_solution_values, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_title(f'Analytical Solution at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Calculate and plot error
    error = np.linalg.norm(predictions - analytical_solution_values) / len(predictions.flatten())
    print(f"Error at t={t} = {error:.2e}")


    ax = axes[3*idx + 2]
    error_contour = ax.contourf(x_grid, y_grid, np.abs(predictions - analytical_solution_values), levels=50, cmap='viridis')
    fig.colorbar(error_contour, ax=ax)
    ax.set_title(f'Error at t={t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

fig.suptitle(f"PINNs solution of monodomain model for $x, y \\in [0,1]$", fontweight='bold')
plt.savefig(f"figures/Comparison_PINNs_epochs_{epochs}.pdf")

# Show plots
plt.tight_layout()
plt.show()


"""
Some results:

Epoch 5000, PDE Loss: 2.8820e-02, IC Loss: 1.8764e-04, BC Loss: 6.2604e-03, Total Loss: 3.5268e-02

Error at t=0.5 = 6.04e-04
Error at t=1 = 9.46e-04

Epoch 10000, PDE Loss: 1.6222e-02, IC Loss: 1.4379e-04, BC Loss: 4.6918e-03, Total Loss: 2.1058e-02
Error at t=0.5 = 1.56e-03
Error at t=1 = 1.40e-03


"""

