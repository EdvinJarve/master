import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

class PINN(nn.Module):
    def __init__(self, num_inputs, num_layers, num_neurons, activation, device):
        super(PINN, self).__init__()
        self.device = device
        layers = [nn.Linear(num_inputs, num_neurons), activation]
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), activation]
        layers += [nn.Linear(num_neurons, 2)]
        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)

class BidomainSolver:
    def __init__(self, Mi, Me, model, device, initial_condition, boundary_condition, I_ion_func):
        self.Mi = Mi
        self.Me = Me
        self.model = model
        self.device = device
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.I_ion_func = I_ion_func
    
    def pde(self, X_collocation, I_ion):
        x, y, t = X_collocation[:, 0:1], X_collocation[:, 1:2], X_collocation[:, 2:3]
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        outputs = self.model(torch.cat([x, y, t], dim=1))
        v, ue = outputs[:, 0:1], outputs[:, 1:2]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        ue_x = torch.autograd.grad(ue, x, grad_outputs=torch.ones_like(ue), create_graph=True)[0]
        ue_y = torch.autograd.grad(ue, y, grad_outputs=torch.ones_like(ue), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        ue_xx = torch.autograd.grad(ue_x, x, grad_outputs=torch.ones_like(ue_x), create_graph=True)[0]
        ue_yy = torch.autograd.grad(ue_y, y, grad_outputs=torch.ones_like(ue_y), create_graph=True)[0]

        laplacian_v = v_xx + v_yy
        laplacian_ue = ue_xx + ue_yy
        residual_1 = v_t - (self.Mi * laplacian_v + self.Mi * laplacian_ue) - I_ion
        residual_2 = self.Mi * laplacian_v + (self.Mi + self.Me) * laplacian_ue

        return residual_1.pow(2).mean() + residual_2.pow(2).mean()
    
    def IC(self, x):
        return self.initial_condition(x, self.model)
    
    def BC_neumann(self, x, normal_vectors):
        return self.boundary_condition(x, self.model, normal_vectors)
    
    def train_step(self, X_boundary, X_collocation, X_ic, optimizer, normal_vectors):
        optimizer.zero_grad()
        I_ion = self.I_ion_func(X_collocation)
        IC_loss = self.IC(X_ic)
        pde_loss = self.pde(X_collocation, I_ion)
        BC_loss = self.BC_neumann(X_boundary, normal_vectors)
        total_loss = IC_loss + BC_loss + pde_loss
        total_loss.backward()
        optimizer.step()
        return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()

    def train(self, X_boundary, X_collocation, X_ic, normal_vectors, epochs, lr, gamma):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        for epoch in range(epochs + 1):
            pde_loss, IC_loss, BC_loss, total_loss = self.train_step(
                X_boundary, X_collocation, X_ic, optimizer, normal_vectors
            )
            if epoch % 100 == 0:
                scheduler.step()
                print(f'Epoch {epoch}, PDE Loss: {pde_loss:.4e}, IC Loss: {IC_loss:.4e}, BC Loss: {BC_loss:.4e}, Total Loss: {total_loss:.4e}')
    
    def create_collocation_points(self, N_space, N_time):
        x_space = np.linspace(0, 1, N_space)
        y_space = np.linspace(0, 1, N_space)
        x_time = np.linspace(0, 1, N_time)
        x_collocation = x_space[1:-1]
        y_collocation = y_space[1:-1]
        t_collocation = x_time[1:-1]
        x_collocation, y_collocation, t_collocation = np.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij')
        X_collocation = np.vstack((x_collocation.flatten(), y_collocation.flatten(), t_collocation.flatten())).T
        return torch.tensor(X_collocation, dtype=torch.float32, device=self.device)

    def create_initial_condition_tensors(self, N_space):
        x_space = np.linspace(0, 1, N_space)
        y_space = np.linspace(0, 1, N_space)
        t_ic = np.zeros((N_space, N_space))
        x_ic, y_ic = np.meshgrid(x_space, y_space, indexing='ij')
        X_ic = np.vstack((x_ic.flatten(), y_ic.flatten(), t_ic.flatten())).T
        return torch.tensor(X_ic, dtype=torch.float32, device=self.device)

    def create_boundary_condition_tensors(self, N_space, N_time):
        x_space = np.linspace(0, 1, N_space)
        y_space = np.linspace(0, 1, N_space)
        x_time = np.linspace(0, 1, N_time)
        x_boundary = np.array([0, 1])
        t_boundary = x_time
        x_boundary, y_boundary, t_boundary = np.meshgrid(x_boundary, y_space, t_boundary, indexing='ij')
        x_side, y_side, t_side = np.meshgrid(x_space, np.array([0, 1]), x_time, indexing='ij')
        X_boundary = np.vstack((np.concatenate([x_boundary.flatten(), x_side.flatten()]), np.concatenate([y_boundary.flatten(), y_side.flatten()]), np.concatenate([t_boundary.flatten(), t_side.flatten()]))).T
        return torch.tensor(X_boundary, dtype=torch.float32, device=self.device)

    def create_normal_vectors(self, X_boundary):
        normal_vectors = np.zeros_like(X_boundary)
        normal_vectors[(X_boundary[:, 0] == 0), 0] = -1
        normal_vectors[(X_boundary[:, 0] == 1), 0] = 1
        normal_vectors[(X_boundary[:, 1] == 0), 1] = -1
        normal_vectors[(X_boundary[:, 1] == 1), 1] = 1
        return torch.tensor(normal_vectors, dtype=torch.float32, device=self.device)
    
    def get_predictions(self, x_grid, y_grid, t):
        x_flat = x_grid.flatten()[:, None]
        y_flat = y_grid.flatten()[:, None]
        t_flat = np.full_like(x_flat, t)
        input_tensor = torch.tensor(np.hstack((x_flat, y_flat, t_flat)), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.model(input_tensor)
        v_predictions = predictions[:, 0].cpu().numpy().reshape(x_grid.shape)
        ue_predictions = predictions[:, 1].cpu().numpy().reshape(x_grid.shape)
        return v_predictions, ue_predictions

from torch.utils.data import DataLoader, TensorDataset

class BatchedBidomainSolver(BidomainSolver):
    def __init__(self, Mi, Me, model, device, initial_condition, boundary_condition, I_ion_func, batch_size=32):
        super().__init__(Mi, Me, model, device, initial_condition, boundary_condition, I_ion_func)
        self.batch_size = batch_size

    def train(self, X_boundary, X_collocation, X_ic, normal_vectors, epochs, lr, gamma):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # Create DataLoaders for batching
        boundary_dataset = TensorDataset(X_boundary, normal_vectors)
        boundary_loader = DataLoader(boundary_dataset, batch_size=self.batch_size, shuffle=True)
        
        collocation_dataset = TensorDataset(X_collocation)
        collocation_loader = DataLoader(collocation_dataset, batch_size=self.batch_size, shuffle=True)
        
        ic_dataset = TensorDataset(X_ic)
        ic_loader = DataLoader(ic_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs + 1):
            total_pde_loss, total_ic_loss, total_bc_loss, total_loss = 0, 0, 0, 0
            
            for (X_bc_batch, normal_vectors_batch), X_coll_batch, X_ic_batch in zip(boundary_loader, collocation_loader, ic_loader):
                pde_loss, IC_loss, BC_loss, batch_total_loss = self.train_step(
                    X_bc_batch, X_coll_batch[0], X_ic_batch[0], optimizer, normal_vectors_batch
                )
                total_pde_loss += pde_loss
                total_ic_loss += IC_loss
                total_bc_loss += BC_loss
                total_loss += batch_total_loss
            
            scheduler.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, PDE Loss: {total_pde_loss:.4e}, IC Loss: {total_ic_loss:.4e}, BC Loss: {total_bc_loss:.4e}, Total Loss: {total_loss:.4e}')



def initial_condition(x, model):
    x_space, y_space, t_time = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    expected_u0 = torch.zeros_like(t_time)
    u0 = model(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))
    return (u0[:, 0:1] - expected_u0).pow(2).mean() + (u0[:, 1:2] - expected_u0).pow(2).mean()

def boundary_condition(x, model, normal_vectors):
    x.requires_grad_(True)
    outputs = model(x)
    v, ue = outputs[:, 0:1], outputs[:, 1:2]
    gradients_v = torch.autograd.grad(outputs=v, inputs=x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    gradients_ue = torch.autograd.grad(outputs=ue, inputs=x, grad_outputs=torch.ones_like(ue), create_graph=True)[0]
    normal_flux_v = torch.sum(gradients_v * normal_vectors, dim=1)
    normal_flux_ue = torch.sum(gradients_ue * normal_vectors, dim=1)
    expected_value_v = torch.zeros_like(normal_flux_v)
    expected_value_ue = torch.zeros_like(normal_flux_ue)
    return (normal_flux_v - expected_value_v).pow(2).mean() + (normal_flux_ue - expected_value_ue).pow(2).mean()

def I_ion_func(X_collocation):
    x, y, t = X_collocation[:, 0:1], X_collocation[:, 1:2], X_collocation[:, 2:3]
    I_ion = torch.cos(t) * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y) + 4 * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y) * torch.sin(t)
    return I_ion

def analytical_solution_v(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    t_coord = x[:, 2:3]
    return np.cos(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord) * np.sin(t_coord)

def analytical_solution_u_e(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    t_coord = x[:, 2:3]
    return -np.cos(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord) * np.sin(t_coord) / 2.0

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_space = 20
    N_time = 20

    model = PINN(num_inputs=3, num_layers=2, num_neurons=32, activation=nn.SiLU(), device=device)
    solver = BidomainSolver(Mi=torch.tensor(1.0), Me=1.0, model=model, device=device, initial_condition=initial_condition, boundary_condition=boundary_condition, I_ion_func=I_ion_func)

    X_collocation = solver.create_collocation_points(N_space, N_time)
    X_ic = solver.create_initial_condition_tensors(N_space)
    X_boundary = solver.create_boundary_condition_tensors(N_space, N_time)
    normal_vectors = solver.create_normal_vectors(X_boundary)

    epochs = 5000
    lr = 1e-2
    gamma = 0.96

    start_time = time.time()
    solver.train(X_boundary, X_collocation, X_ic, normal_vectors, epochs, lr, gamma)
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Computation time (excluding visualization): {computation_time:.2f} seconds")

    # Visualization
    x_space = np.linspace(0, 1, N_space)
    y_space = np.linspace(0, 1, N_space)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    time_point = 1.0

    predictions_v, predictions_u_e = solver.get_predictions(x_grid, y_grid, time_point)

    x_flat = x_grid.flatten()[:, None]
    y_flat = y_grid.flatten()[:, None]
    t_flat = np.full_like(x_flat, time_point)
    input_tensor = np.hstack((x_flat, y_flat, t_flat))

    analytical_solution_values_v = analytical_solution_v(input_tensor).reshape(x_grid.shape)
    analytical_solution_values_u_e = analytical_solution_u_e(input_tensor).reshape(x_grid.shape)

    # Plot numerical solution for v
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax = axes[0]
    contour_v = ax.contourf(x_grid, y_grid, predictions_v, levels=50, cmap='viridis')
    fig.colorbar(contour_v, ax=ax)
    ax.set_title(f'Numerical Solution for $v$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot analytical solution for v
    ax = axes[1]
    contour_v_analytical = ax.contourf(x_grid, y_grid, analytical_solution_values_v, levels=50, cmap='viridis')
    fig.colorbar(contour_v_analytical, ax=ax)
    ax.set_title(f'Analytical Solution for $v$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"figures_bi_pinns/Comparison_PINNs_bidomain_v_t_{time_point}.pdf")
    plt.show()

    # Prepare for plotting numerical and analytical solutions for u_e
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot numerical solution for u_e
    ax = axes[0]
    contour_u_e = ax.contourf(x_grid, y_grid, predictions_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour_u_e, ax=ax)
    ax.set_title(f'Numerical Solution for $u_e$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot analytical solution for u_e
    ax = axes[1]
    contour_u_e_analytical = ax.contourf(x_grid, y_grid, analytical_solution_values_u_e, levels=50, cmap='viridis')
    fig.colorbar(contour_u_e_analytical, ax=ax)
    ax.set_title(f'Analytical Solution for $u_e$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"figures_bi_pinns/Comparison_PINNs_bidomain_u_e_t_{time_point}.pdf")
    plt.show()

    # Plot error for v
    fig, ax = plt.subplots(figsize=(11, 6))
    error_contour_v = ax.contourf(x_grid, y_grid, np.abs(predictions_v - analytical_solution_values_v), levels=50, cmap='viridis')
    fig.colorbar(error_contour_v, ax=ax)
    ax.set_title(f'Error for $v$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"figures_bi_pinns/Error_PINNs_bidomain_v_{time_point}.pdf")
    plt.show()

    # Plot error for u_e
    fig, ax = plt.subplots(figsize=(11, 6))
    error_contour_u_e = ax.contourf(x_grid, y_grid, np.abs(predictions_u_e - analytical_solution_values_u_e), levels=50, cmap='viridis')
    fig.colorbar(error_contour_u_e, ax=ax)
    ax.set_title(f'Error for $u_e$ at $t={time_point}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(f"figures_bi_pinns/Error_PINNs_bidomain_u_e_t_{time_point}.pdf")
    plt.show()

if __name__ == "__main__":
    main()
