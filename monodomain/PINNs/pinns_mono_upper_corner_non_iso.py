import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib as mpl
import time
import os 

"""
This code uses PINNs to solve the 2D monodomain model. It handles a corner stimulation from t = 0 to t = 0.2 
with an amplitude of 200. The conductivity tensor is set to be linear, making the current downward linearly.
"""


# Create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

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
        layers += [nn.Linear(num_neurons, 1)]  # Output layer

        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)
    
def stimulus_expression(t, duration=0.1, current_value = 50 ) :
    spatial_mask = (x <= 0.2) & (y >= 0.8)
    source = torch.where((spatial_mask) & (t < duration), 
                         torch.tensor(current_value, device=x.device, dtype=x.dtype), 
                         torch.tensor(0.0, device=x.device, dtype=x.dtype))
    return source

def pde(X_collocation, model, M_i):
    x = X_collocation[:, 0:1]
    y = X_collocation[:, 1:2]
    t = X_collocation[:, 2:3]

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    u = model(torch.cat([x, y, t], dim=1))

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    source = source_term(x, y, t)
    laplacian_u = M_i[0, 0] * u_xx + M_i[1, 1] * u_yy
    residual = u_t - laplacian_u - source

    return residual.pow(2).mean()

def IC(x, model):
    x_space = x[:, 0:1]
    y_space = x[:, 1:2]
    t_time = x[:, 2:3]

    expected_u0 = torch.zeros_like(t_time)
    u0 = model(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))

    return (u0 - expected_u0).pow(2).mean()

def BC_neumann(x, model, normal_vectors):
    x.requires_grad_(True)
    u = model(x)

    gradients = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    normal_flux = torch.sum(gradients * normal_vectors, dim=1)

    expected_value = torch.zeros_like(normal_flux)
    return (normal_flux - expected_value).pow(2).mean()

def train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors, M_i):
    optimizer.zero_grad()

    IC_loss = IC(X_ic, model)
    pde_loss = pde(X_collocation, model, M_i)
    BC_loss = BC_neumann(X_boundary, model, normal_vectors)

    total_loss = IC_loss + BC_loss + pde_loss
    total_loss.backward()
    optimizer.step()

    return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()

# Define number of points in the spatial and temporal domains
N_space = 40
N_time = 40

x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_time = np.linspace(0, 1, N_time)

x_collocation, y_collocation, t_collocation = np.meshgrid(x_space[1:-1], y_space[1:-1], x_time[1:-1], indexing='ij')
x_collocation = x_collocation.flatten()
y_collocation = y_collocation.flatten()
t_collocation = t_collocation.flatten()
X_collocation = np.vstack((x_collocation, y_collocation, t_collocation)).T

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_collocation = torch.tensor(X_collocation, dtype=torch.float32, device=device)

t_ic = np.zeros((N_space, N_space))
x_ic, y_ic = np.meshgrid(x_space, y_space, indexing='ij')
x_ic = x_ic.flatten()
y_ic = y_ic.flatten()
t_ic = t_ic.flatten()
X_ic = np.vstack((x_ic, y_ic, t_ic)).T
X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)

x_boundary = np.array([0, 1])
t_boundary = x_time
x_boundary, y_boundary, t_boundary = np.meshgrid(x_boundary, y_space, t_boundary, indexing='ij')
x_boundary = x_boundary.flatten()
y_boundary = y_boundary.flatten()
t_boundary = t_boundary.flatten()

y_boundary_side = np.array([0, 1])
x_side, y_side, t_side = np.meshgrid(x_space, y_boundary_side, x_time, indexing='ij')
x_side = x_side.flatten()
y_side = y_side.flatten()
t_side = t_side.flatten()

x_boundary_all = np.concatenate([x_boundary, x_side])
y_boundary_all = np.concatenate([y_boundary, y_side])
t_boundary_all = np.concatenate([t_boundary, t_side])

X_boundary = np.vstack((x_boundary_all, y_boundary_all, t_boundary_all)).T
X_boundary = torch.tensor(X_boundary, dtype=torch.float32, device=device)

normal_vectors = np.zeros_like(X_boundary)
x_coords = X_boundary[:, 0]
y_coords = X_boundary[:, 1]

normal_vectors[(x_coords == 0), 0] = -1
normal_vectors[(x_coords == 1), 0] = 1
normal_vectors[(y_coords == 0), 1] = -1
normal_vectors[(y_coords == 1), 1] = 1

normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32, device=device)

X_collocation = X_collocation.to(device)
X_ic = X_ic.to(device)
X_boundary = X_boundary.to(device)
normal_vectors = normal_vectors.to(device)

theta = torch.tensor(torch.pi/4, dtype=torch.float32, device=device)
cos_theta = torch.cos(theta)
sin_theta = torch.sin(theta)
R = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=torch.float32, device=device)

D_i = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=device)
M_i = R.T @ D_i @ R

model = PINN(num_inputs=3, num_layers=4, num_neurons=32, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

epochs = 5000

loss_list = []
epoch_list = []

start_time = time.time()
for epoch in range(epochs+1):
    pde_loss, IC_loss, BC_loss, total_loss = train_step(X_boundary, X_collocation, X_ic, optimizer, model, normal_vectors, M_i)
    
    if epoch % 100 == 0:
        loss_list.append(total_loss)
        epoch_list.append(epoch)
        scheduler.step()
        print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')

end_time = time.time()
computation_time = end_time - start_time
print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig(f"figures/pinns_non_iso_loss_for_epochs={epochs}_corner.pdf")
plt.close()

x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_grid, y_grid = np.meshgrid(x_space, y_space)

x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

time_points = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0]

solutions = {}

for t in time_points:
    t_array = np.full_like(x_flat, fill_value=t)
    model_input = np.stack((x_flat, y_flat, t_array), axis=-1)
    model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(model_input_tensor).cpu().numpy().flatten()
        
    solutions[t] = predictions

pairwise_time_points = [(0.0, 0.1), (0.2, 0.3), (0.6, 1.0)]

for pair in pairwise_time_points:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    for idx, t_eval in enumerate(pair):
        numerical_solution = solutions[t_eval]
        
        ax = axes[idx]
        contour = ax.tricontourf(x_flat, y_flat, numerical_solution, levels=50, cmap='viridis', vmin=0, vmax=0.2)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f'Numerical Solution at t={t_eval}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"figures_pinns_mono/Current_non_iso_PINNs_solution_t{pair[0]}_t{pair[1]}.pdf")
    plt.show()

"""
Some results:






"""

