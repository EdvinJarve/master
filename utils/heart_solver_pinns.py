import numpy as np
import torch
import torch.nn as nn
import time

class PINN(nn.Module):
    """
    Base class for Physics-Informed Neural Networks (PINNs).

    Attributes:
        model (nn.Sequential): Neural network model.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, device):
        """
        Initialize the PINN.

        Parameters:
            num_inputs (int): Number of input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            num_outputs (int): Number of output features.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(PINN, self).__init__()
        self.device = device
        activation = nn.SiLU()

        layers = [nn.Linear(num_inputs, num_neurons), activation]  # Input layer
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), activation]  # Hidden layers
        layers += [nn.Linear(num_neurons, num_outputs)]  # Output layer

        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class MonodomainSolverPINNs(PINN):
    """
    PINN solver for the monodomain model.

    Attributes:
        source_term_func (callable): Function representing the source term in the PDE.
        M (float or np.ndarray): Conductivity tensor.
    """
    def __init__(self, num_inputs, num_layers, num_neurons, device, source_term_func, M):
        """
        Initialize the monodomain solver.

        Parameters:
            num_inputs (int): Number of input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            device (str): Device to run the model on ('cpu' or 'cuda').
            source_term_func (callable): Function representing the source term in the PDE.
            M (float or np.ndarray): Conductivity tensor.
        """
        super(MonodomainSolverPINNs, self).__init__(num_inputs, num_layers, num_neurons, 1, device)  # 1 output
        self.source_term_func = source_term_func
        self.M = M

    
    def prepare_data(self, N_space, N_time):
        """
        Prepare the training data for the monodomain solver.

        Parameters:
            N_space (int): Number of spatial points.
            N_time (int): Number of temporal points.
        """
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
        self.X_collocation = torch.tensor(X_collocation, dtype=torch.float32, device=self.device)

        # Define the initial condition coordinates (spatial grid at t=0)
        t_ic = np.zeros((N_space, N_space))
        x_ic, y_ic = np.meshgrid(x_space, y_space, indexing='ij')
        x_ic = x_ic.flatten()
        y_ic = y_ic.flatten()
        t_ic = t_ic.flatten()

        # Combine the spatial and temporal coordinates
        X_ic = np.vstack((x_ic, y_ic, t_ic)).T
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32, device=self.device)

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
        self.X_boundary = torch.tensor(X_boundary, dtype=torch.float32, device=self.device)

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

        self.normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32, device=self.device)

    
    def pde(self, X_collocation):
        """
        Define the partial differential equation for the monodomain model.

        Parameters:
            X_collocation (torch.Tensor): Collocation points.

        Returns:
            torch.Tensor: Residual loss.
        """
        x = X_collocation[:, 0:1]  # X spatial coordinates
        y = X_collocation[:, 1:2]  # Y spatial coordinates
        t = X_collocation[:, 2:3]  # Time coordinates

        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        outputs = self.model(torch.cat([x, y, t], dim=1))
        v = outputs[:, 0:1]

        # First derivatives
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        # Gradient vector
        grad_v = torch.cat([v_x, v_y], dim=1)

        # Ensure Mi is a tensor
        self.M = torch.tensor(self.M, dtype=torch.float32, device=x.device) if isinstance(self.M, (float, int, np.ndarray)) else self.M

        if self.M.ndim == 0:  # Scalar case
            # Second derivatives for scalar case
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
            laplacian_v = self.M * (v_xx + v_yy)
        else:  # Tensor case
            Mi_grad_v = grad_v @ self.M

            # Divergence of Mi * grad(v)
            div_Mi_grad_v = torch.autograd.grad(Mi_grad_v[:, 0], x, grad_outputs=torch.ones_like(Mi_grad_v[:, 0]), create_graph=True)[0] + \
                            torch.autograd.grad(Mi_grad_v[:, 1], y, grad_outputs=torch.ones_like(Mi_grad_v[:, 1]), create_graph=True)[0]
            laplacian_v = div_Mi_grad_v

        # Ionic current source term
        I_ion = self.source_term_func(x, y, t)

        # Residual for the monodomain equation
        residual = v_t - laplacian_v - I_ion

        # Combined residual
        residual_loss = residual.pow(2).mean()

        return residual_loss
    
    def IC(self, x):
        """
        Define the initial condition for the monodomain model.

        Parameters:
            x (torch.Tensor): Initial condition points.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        x_space = x[:, 0:1]  # X spatial coordinates
        y_space = x[:, 1:2]  # Y spatial coordinates
        t_time = x[:, 2:3]  # Time coordinate

        # Create a tensor of zeros with the same shape as t_time to represent initial condition
        expected_u0 = torch.zeros_like(t_time)
        # Evaluate the model at the initial time
        u0 = self.forward(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1))

        # Calculate the squared error between the predicted and expected initial condition
        return (u0 - expected_u0).pow(2).mean()
    
    def BC_neumann(self, x, normal_vectors):
        """
        Define the Neumann boundary condition for the monodomain model.

        Parameters:
            x (torch.Tensor): Boundary points.
            normal_vectors (torch.Tensor): Normal vectors at boundary points.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        x.requires_grad_(True)
        u = self.forward(x)

        # Compute gradients of u with respect to spatial coordinates
        gradients = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        normal_flux = torch.sum(gradients * normal_vectors, dim=1)

        expected_value = torch.zeros_like(normal_flux)
        return (normal_flux - expected_value).pow(2).mean()
    
    def train_step(self, optimizer):
        """
        Perform a single training step.

        Parameters:
            optimizer (torch.optim.Optimizer): Optimizer for training.

        Returns:
            tuple: PDE loss, initial condition loss, boundary condition loss, and total loss.
        """
        optimizer.zero_grad()

        IC_loss = self.IC(self.X_ic)
        pde_loss = self.pde(self.X_collocation)
        BC_loss = self.BC_neumann(self.X_boundary, self.normal_vectors)

        total_loss = IC_loss + BC_loss + pde_loss
        total_loss.backward()
        optimizer.step()

        return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()
    
class BidomainSolverPINNs(PINN):
    """
    PINN solver for the bidomain model.

    Attributes:
        source_term_func (callable): Function for the source term.
        Mi (torch.Tensor or float): Intracellular conductivity tensor or scalar.
        Me (torch.Tensor or float): Extracellular conductivity tensor or scalar.
    """
    def __init__(self, num_inputs, num_layers, num_neurons, device, source_term_func, Mi, Me):
        """
        Initialize the BidomainSolverPINNs class.

        Args:
            num_inputs (int): Number of input features.
            num_layers (int): Number of layers in the neural network.
            num_neurons (int): Number of neurons per layer.
            device (torch.device): Device to run the model on (CPU or GPU).
            source_term_func (callable): Function to compute the source term.
            Mi (torch.Tensor or float): Intracellular conductivity tensor or scalar.
            Me (torch.Tensor or float): Extracellular conductivity tensor or scalar.
        """
        super(BidomainSolverPINNs, self).__init__(num_inputs, num_layers, num_neurons, 2, device)
        self.source_term_func = source_term_func
        self.Mi = Mi
        self.Me = Me

    def prepare_data(self, N_space, N_time):
        """
        Prepare training data by defining spatial and temporal coordinates.

        Args:
            N_space (int): Number of spatial points.
            N_time (int): Number of temporal points.
        """
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
        self.X_collocation = torch.tensor(X_collocation, dtype=torch.float32, device=self.device)

        # Define the initial condition coordinates (spatial grid at t=0)
        t_ic = np.zeros((N_space, N_space))
        x_ic, y_ic = np.meshgrid(x_space, y_space, indexing='ij')
        x_ic = x_ic.flatten()
        y_ic = y_ic.flatten()
        t_ic = t_ic.flatten()

        # Combine the spatial and temporal coordinates
        X_ic = np.vstack((x_ic, y_ic, t_ic)).T
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32, device=self.device)

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
        self.X_boundary = torch.tensor(X_boundary, dtype=torch.float32, device=self.device)

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

        self.normal_vectors = torch.tensor(normal_vectors, dtype=torch.float32, device=self.device)

    def pde(self, X_collocation):
        """
        Compute the PDE residual for the bidomain model.

        Args:
            X_collocation (torch.Tensor): Collocation points for evaluating the PDE.

        Returns:
            torch.Tensor: Residual loss of the PDE.
        """
        x = X_collocation[:, 0:1]  # X spatial coordinates
        y = X_collocation[:, 1:2]  # Y spatial coordinates
        t = X_collocation[:, 2:3]  # Time coordinates

        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        outputs = self.model(torch.cat([x, y, t], dim=1))
        v = outputs[:, 0:1]
        ue = outputs[:, 1:2]

        # First derivatives
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        ue_x = torch.autograd.grad(ue, x, grad_outputs=torch.ones_like(ue), create_graph=True, retain_graph=True)[0]
        ue_y = torch.autograd.grad(ue, y, grad_outputs=torch.ones_like(ue), create_graph=True, retain_graph=True)[0]

        # Gradient vectors
        grad_v = torch.cat([v_x, v_y], dim=1)
        grad_ue = torch.cat([ue_x, ue_y], dim=1)

        # Ensure Mi and Me are tensors
        self.Mi = torch.tensor(self.Mi, dtype=torch.float32, device=x.device) if isinstance(self.Mi, (float, int, np.ndarray)) else self.Mi
        self.Me = torch.tensor(self.Me, dtype=torch.float32, device=x.device) if isinstance(self.Me, (float, int, np.ndarray)) else self.Me

        if self.Mi.ndim == 0:  # Scalar case
            # Laplacians for scalar case
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
            ue_xx = torch.autograd.grad(ue_x, x, grad_outputs=torch.ones_like(ue_x), create_graph=True)[0]
            ue_yy = torch.autograd.grad(ue_y, y, grad_outputs=torch.ones_like(ue_y), create_graph=True)[0]

            laplacian_v = self.Mi * (v_xx + v_yy)
            laplacian_ue = self.Mi * (ue_xx + ue_yy)
        else:  # Tensor case
            M = torch.tensor([[-1, 1], [1, -1]], dtype=torch.float32, device=x.device)
            Mi_grad_v = grad_v @ M
            Mi_grad_ue = grad_ue @ M

            # Divergence of Mi * grad(v) and Mi * grad(ue)
            div_Mi_grad_v = torch.autograd.grad(Mi_grad_v[:, 0], x, grad_outputs=torch.ones_like(Mi_grad_v[:, 0]), create_graph=True)[0] + \
                            torch.autograd.grad(Mi_grad_v[:, 1], y, grad_outputs=torch.ones_like(Mi_grad_v[:, 1]), create_graph=True)[0]
            div_Mi_grad_ue = torch.autograd.grad(Mi_grad_ue[:, 0], x, grad_outputs=torch.ones_like(Mi_grad_ue[:, 0]), create_graph=True)[0] + \
                            torch.autograd.grad(Mi_grad_ue[:, 1], y, grad_outputs=torch.ones_like(Mi_grad_ue[:, 1]), create_graph=True)[0]

            laplacian_v = div_Mi_grad_v
            laplacian_ue = div_Mi_grad_ue

        # Ionic current source term
        I_ion = self.source_term_func(x, y, t)

        # Residuals for the bidomain equations
        if self.Mi.ndim == 0:  # Scalar case
            residual_1 = v_t - (laplacian_v + laplacian_ue) - I_ion
            residual_2 = self.Mi * (v_xx + v_yy) + (self.Mi + self.Me) * (ue_xx + ue_yy)
        else:  # Tensor case
            residual_1 = v_t - (laplacian_v + laplacian_ue) - I_ion
            M_ue = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32, device=x.device)
            Me_grad_ue = grad_ue @ (M + M_ue)
            div_Me_grad_ue = torch.autograd.grad(Me_grad_ue[:, 0], x, grad_outputs=torch.ones_like(Me_grad_ue[:, 0]), create_graph=True)[0] + \
                            torch.autograd.grad(Me_grad_ue[:, 1], y, grad_outputs=torch.ones_like(Me_grad_ue[:, 1]), create_graph=True)[0]

            residual_2 = div_Mi_grad_v + div_Me_grad_ue

        # Combined residual
        residual = residual_1.pow(2).mean() + residual_2.pow(2).mean()

        return residual

    def IC(self, x):
        """
        Compute the initial condition loss.

        Args:
            x (torch.Tensor): Initial condition points.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        x_space = x[:, 0:1]  # X spatial coordinates
        y_space = x[:, 1:2]  # Y spatial coordinates
        t_time = x[:, 2:3]  # Time coordinate

        # Create a tensor of zeros with the same shape as t_time to represent initial condition
        expected_u0 = torch.zeros_like(t_time)
        # Evaluate the model at the initial time
        u0 = self.forward(torch.cat((x_space, y_space, torch.zeros_like(t_time)), dim=1)) #self.model

        # Calculate the squared error between the predicted and expected initial condition
        return (u0[:, 0:1] - expected_u0).pow(2).mean() + (u0[:, 1:2] - expected_u0).pow(2).mean()

    def BC_neumann(self, x, normal_vectors):
        """
        Compute the Neumann boundary condition loss.

        Args:
            x (torch.Tensor): Boundary points.
            normal_vectors (torch.Tensor): Normal vectors at the boundary points.

        Returns:
            torch.Tensor: Neumann boundary condition loss.
        """
        x.requires_grad_(True)
        outputs = self.model(x)
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

    def train_step(self, optimizer):
        """
        Perform a single training step.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for training.

        Returns:
            tuple: Tuple containing the PDE loss, initial condition loss, boundary condition loss, and total loss.
        """
        optimizer.zero_grad()

        IC_loss = self.IC(self.X_ic)
        pde_loss = self.pde(self.X_collocation)
        BC_loss = self.BC_neumann(self.X_boundary, self.normal_vectors)

        total_loss = IC_loss + BC_loss + pde_loss
        total_loss.backward()
        optimizer.step()

        return pde_loss.item(), IC_loss.item(), BC_loss.item(), total_loss.item()
