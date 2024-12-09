import torch
import torch.nn as nn
import numpy as np
import time 
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, List, Any
import torch.nn.functional as F

class PINN(nn.Module):
    """
    Base class for Physics-Informed Neural Networks (PINNs).
    """
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, device):
        super(PINN, self).__init__()
        self.device = device
        activation = nn.Tanh()
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
        return self.model(x)

class MonodomainSolverPINNs(PINN):
    """
    PINN solver for the monodomain model with support for explicit source terms or ODE-governed current terms.
    """
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        device: str,
        source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        M: Union[float, int, list, np.ndarray, torch.Tensor] = None,
        use_ode: bool = False,
        ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_state_vars: int = 0,
        loss_function: str = 'L2',
        loss_weights: Optional[Dict[str, float]] = None,
        weight_strategy: str = 'manual',  # 'manual' or 'dynamic'
        alpha: float = 0.9,  # moving average parameter for dynamic weights
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0
    ):
        """
        Initializes the MonodomainSolverPINNs.
        """
        if use_ode:
            output_dimension = 1 + n_state_vars
        else:
            output_dimension = 1

        super(MonodomainSolverPINNs, self).__init__(num_inputs, num_layers, num_neurons, output_dimension, device)
        self.source_term_func = source_term_func

        if isinstance(M, (float, int, list, np.ndarray)):
            self.M = torch.tensor(M, dtype=torch.float32, device=device)
        elif isinstance(M, torch.Tensor):
            self.M = M.to(device)
        else:
            raise TypeError("M must be a float, int, list, np.ndarray, or torch.Tensor.")

        self.is_scalar_M = self.M.ndim == 0
        self.use_ode = use_ode
        self.ode_func = ode_func
        self.n_state_vars = n_state_vars
        self.loss_function = loss_function

        # Initialize loss weights based on strategy
        self.weight_strategy = weight_strategy
        self.alpha = alpha
        
        if weight_strategy == 'manual':
            if loss_weights is None:
                self.loss_weights = {
                    'pde_loss': 1.0,
                    'IC_loss': 1.0,
                    'BC_loss': 1.0,
                    'data_loss': 1.0,
                    'ode_loss': 1.0
                }
            else:
                self.loss_weights = loss_weights
        else:  # dynamic
            # Initialize dynamic weights
            self.lambda_ic = torch.tensor(1.0, device=device)
            self.lambda_bc = torch.tensor(1.0, device=device)
            self.lambda_r = torch.tensor(1.0, device=device)

        # Initialize data points and expected outputs for data loss
        self.X_data = None
        self.expected_data = None

        # Initialize loss function object for data loss
        if self.loss_function == 'L2':
            self.loss_function_obj = nn.MSELoss()
        elif self.loss_function == 'L1':
            self.loss_function_obj = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def pde(self, X_collocation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the PDE residual loss and ODE residual loss (if applicable).
        """
        # Input is already normalized, no need to scale
        X_collocation.requires_grad_(True)
        
        # Forward pass through the network
        outputs = self.forward(X_collocation)
        
        # Extract u and compute gradients
        if self.use_ode:
            u = outputs[:, 0:1]  # Shape: (N_collocation, 1)
            w = outputs[:, 1:]   # Shape: (N_collocation, n_state_vars)
        else:
            u = outputs  # Shape: (N_collocation, 1)
        
        # Compute gradients
        du_dx = torch.autograd.grad(u.sum(), X_collocation, create_graph=True)[0]
        d2u_dx2 = torch.zeros_like(u)
        
        # Second derivatives for x and y
        d2u_dx2_x = torch.autograd.grad(du_dx[:, 0].sum(), X_collocation, create_graph=True)[0][:, 0:1]
        d2u_dx2_y = torch.autograd.grad(du_dx[:, 1].sum(), X_collocation, create_graph=True)[0][:, 1:2]
        
        # Time derivative
        du_dt = du_dx[:, 2:3]
        
        # Compute Laplacian term
        if self.is_scalar_M:
            laplacian_term = self.M * (d2u_dx2_x + d2u_dx2_y)
        else:
            # For tensor conductivity
            laplacian_term = (
                self.M[0] * d2u_dx2_x +
                self.M[1] * d2u_dx2_y
            )
        
        # Compute source term
        if self.use_ode:
            # Use ODE for current terms
            ode_residuals = self.ode_func(u, w, X_collocation)
            source_term = ode_residuals[:, 0:1]  # First residual corresponds to u
            ode_loss = torch.mean(torch.square(ode_residuals[:, 1:]))  # Remaining residuals
        else:
            # Use explicit source term function
            source_term = self.source_term_func(X_collocation[:, :2], X_collocation[:, 2:3])
            ode_loss = torch.tensor(0.0, device=self.device)
        
        # Compute PDE residual: du/dt - div(M * grad(u)) = I_ion
        pde_residual = du_dt - laplacian_term - source_term
        
        # Compute mean squared PDE residual
        pde_loss = torch.mean(torch.square(pde_residual))
        
        return pde_loss, ode_loss

    def IC(self, X_ic: torch.Tensor, expected_u0: torch.Tensor) -> torch.Tensor:
        """
        Compute the Initial Condition (IC) loss.
        """
        # The input X_ic is already normalized, no need to scale it
        X_ic.requires_grad_(True)
        
        # Forward pass through the network with scaled input
        outputs = self.forward(X_ic)
        
        # Extract u based on whether ODE is used
        if self.use_ode:
            u0 = outputs[:, 0:1]  # Assuming u0 is the first output
        else:
            u0 = outputs  # Shape: (N, 1)
        
        # Compute loss using unscaled values
        if self.loss_function == 'L2':
            IC_loss = nn.MSELoss()(u0, expected_u0.to(self.device))
        elif self.loss_function == 'L1':
            IC_loss = nn.L1Loss()(u0, expected_u0.to(self.device))
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        return IC_loss

    def BC_neumann(
        self,
        X_boundary: torch.Tensor,
        normal_vectors: torch.Tensor,
        expected_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Neumann Boundary Condition (BC) loss.
        """
        # Input is already normalized, no need to scale
        X_boundary.requires_grad_(True)
        
        # Forward pass through the network
        outputs = self.forward(X_boundary)
        
        # Extract u
        if self.use_ode:
            u = outputs[:, 0:1]  # Shape: (N_BC, 1)
        else:
            u = outputs  # Shape: (N_BC, 1)
        
        # Compute spatial derivatives
        du_dx = torch.autograd.grad(u.sum(), X_boundary, create_graph=True)[0][:, :2]
        
        # Compute normal derivative
        if self.is_scalar_M:
            normal_deriv = self.M * torch.sum(du_dx * normal_vectors, dim=1, keepdim=True)
        else:
            # For tensor conductivity
            normal_deriv = torch.sum(
                torch.stack([
                    self.M[0] * du_dx[:, 0],
                    self.M[1] * du_dx[:, 1]
                ]) * normal_vectors.t(),
                dim=0,
                keepdim=True
            ).t()
        
        # If no expected value is provided, assume zero Neumann BC
        if expected_value is None:
            expected_value = torch.zeros_like(normal_deriv)
        
        # Compute BC loss
        bc_loss = torch.mean(torch.square(normal_deriv - expected_value))
        
        return bc_loss

    def data_loss(self, X_data: torch.Tensor, expected_data: torch.Tensor) -> torch.Tensor:
        """
        Compute the data loss between the model's predictions and expected data.
        """
        self.train()
        outputs = self.forward(X_data)
        
        # Extract u based on whether ODE is used
        if self.use_ode:
            u = outputs[:, 0:1]  # Assuming u is the first output
        else:
            u = outputs  # Shape: (N, 1)
        
        if self.loss_function == 'L2':
            loss = nn.MSELoss()(u, expected_data.to(self.device))
        elif self.loss_function == 'L1':
            loss = nn.L1Loss()(u, expected_data.to(self.device))
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        return loss

    def train_step(self, optimizer: torch.optim.Optimizer, batch_size: Optional[int] = None) -> tuple:
        """
        Perform a single training step.
        """
        self.train()
        optimizer.zero_grad()

        # Get mini-batches if batch_size is specified
        if batch_size is not None:
            # Generate separate indices for each dataset since they might have different sizes
            collocation_indices = torch.randperm(self.X_collocation.size(0))[:batch_size]
            ic_indices = torch.randperm(self.X_ic.size(0))[:batch_size]
            boundary_indices = torch.randperm(self.X_boundary.size(0))[:batch_size]

            X_collocation_batch = self.X_collocation[collocation_indices]
            X_ic_batch = self.X_ic[ic_indices]
            expected_u0_batch = self.expected_u0[ic_indices]
            X_boundary_batch = self.X_boundary[boundary_indices]
            normal_vectors_batch = self.normal_vectors[boundary_indices]

            if self.X_data is not None:
                data_indices = torch.randperm(self.X_data.size(0))[:batch_size]
                X_data_batch = self.X_data[data_indices]
                expected_data_batch = self.expected_data[data_indices]
            else:
                X_data_batch = None
                expected_data_batch = None
        else:
            X_collocation_batch = self.X_collocation
            X_ic_batch = self.X_ic
            expected_u0_batch = self.expected_u0
            X_boundary_batch = self.X_boundary
            normal_vectors_batch = self.normal_vectors
            X_data_batch = self.X_data
            expected_data_batch = self.expected_data

        # Update weights if using dynamic strategy
        if self.weight_strategy == 'dynamic':
            self.update_weights(
                X_collocation_batch,
                X_ic_batch,
                expected_u0_batch,
                X_boundary_batch,
                normal_vectors_batch
            )

        # Compute losses
        IC_loss = self.IC(X_ic_batch, expected_u0_batch)
        BC_loss = self.BC_neumann(X_boundary_batch, normal_vectors_batch)
        pde_loss, ode_loss = self.pde(X_collocation_batch)

        # Apply loss weights and aggregate
        if self.weight_strategy == 'manual':
            total_loss = (
                self.loss_weights['IC_loss'] * IC_loss +
                self.loss_weights['BC_loss'] * BC_loss +
                self.loss_weights['pde_loss'] * pde_loss +
                self.loss_weights['ode_loss'] * ode_loss
            )
        else:  # dynamic
            total_loss = (
                self.lambda_ic * IC_loss +
                self.lambda_bc * BC_loss +
                self.lambda_r * pde_loss +
                ode_loss
            )

        # Compute data loss if applicable
        if X_data_batch is not None and expected_data_batch is not None:
            data_loss = self.data_loss(X_data_batch, expected_data_batch)
            if self.weight_strategy == 'manual':
                total_loss += self.loss_weights['data_loss'] * data_loss
            else:  # dynamic
                total_loss += data_loss
        else:
            data_loss = torch.tensor(0.0, device=self.device)

        # Backpropagation with retain_graph=True
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # Return loss values
        return pde_loss.item(), IC_loss.item(), BC_loss.item(), data_loss.item(), ode_loss.item(), total_loss.item()

    def evaluate(self, X_test: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple]:
        """
        Evaluate the trained model on test data.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(X_test.to(self.device))
        
        # Extract u based on whether ODE is used
        if self.use_ode:
            y_pred = outputs[:, 0:1]  # Only u
        else:
            y_pred = outputs  # Shape: (N, 1)

        if y_true is not None:
            if self.loss_function == 'L2':
                loss = nn.MSELoss()(y_pred, y_true.to(self.device)).item()
            elif self.loss_function == 'L1':
                loss = nn.L1Loss()(y_pred, y_true.to(self.device)).item()
            else:
                raise ValueError(f"Unknown loss function: {self.loss_function}")
            return y_pred.cpu(), loss
        else:
            return y_pred.cpu()


    def validate(self, X_collocation_val, X_ic_val, expected_u0_val, X_boundary_val, normal_vectors_val):
        self.eval()
        with torch.enable_grad():  # Enable gradients only within this block
            # Compute PDE Loss
            pde_loss_val, ode_loss_val = self.pde(X_collocation_val)
            
            # Compute IC Loss
            ic_loss_val = self.IC(X_ic_val, expected_u0_val)
            
            # Compute BC Loss
            bc_loss_val = self.BC_neumann(X_boundary_val, normal_vectors_val)
            
            # Total Validation Loss
            if self.weight_strategy == 'manual':
                total_val_loss = (
                    self.loss_weights['pde_loss'] * pde_loss_val +
                    self.loss_weights['IC_loss'] * ic_loss_val +
                    self.loss_weights['BC_loss'] * bc_loss_val
                )
            else:  # dynamic
                total_val_loss = (
                    self.lambda_ic * ic_loss_val +
                    self.lambda_bc * bc_loss_val +
                    self.lambda_r * pde_loss_val +
                    ode_loss_val
                )
        
        return total_val_loss.item()


    def save_model(self, file_path):
        """
        Save the trained model's state dictionary to a file.
        """
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load the model's state dictionary from a file.
        """
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)
        self.eval()
        print(f"Model loaded from {file_path}")

    def compute_gradient_norms(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        """Compute L2 norms of gradients for each loss component."""
        self.zero_grad()
        
        # Compute losses
        pde_loss, _ = self.pde(X_collocation)
        ic_loss = self.IC(X_ic, expected_u0)
        bc_loss = self.BC_neumann(X_boundary, normal_vectors)
        
        # Compute gradients with allow_unused=True and retain_graph=True
        grad_r = torch.autograd.grad(pde_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_ic = torch.autograd.grad(ic_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_bc = torch.autograd.grad(bc_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        
        # Filter out None gradients and compute L2 norms, then detach
        norm_r = torch.sqrt(sum(torch.sum(g**2) for g in grad_r if g is not None)).detach()
        norm_ic = torch.sqrt(sum(torch.sum(g**2) for g in grad_ic if g is not None)).detach()
        norm_bc = torch.sqrt(sum(torch.sum(g**2) for g in grad_bc if g is not None)).detach()
        
        return norm_r, norm_ic, norm_bc

    def update_weights(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        """Update weights using gradient norms and moving average."""
        if self.weight_strategy != 'dynamic':
            return
            
        # Compute gradient norms
        norm_r, norm_ic, norm_bc = self.compute_gradient_norms(
            X_collocation, X_ic, expected_u0, X_boundary, normal_vectors
        )
        
        # Compute sum of norms
        sum_norms = (norm_r + norm_ic + norm_bc).detach()
        
        # Compute new weights
        lambda_ic_new = (sum_norms / (norm_ic + 1e-8)).detach()  # Add small epsilon to prevent division by zero
        lambda_bc_new = (sum_norms / (norm_bc + 1e-8)).detach()
        lambda_r_new = (sum_norms / (norm_r + 1e-8)).detach()
        
        # Update weights using moving average
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * lambda_ic_new
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * lambda_bc_new
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * lambda_r_new

# ====================================================================================
# INVERSE PROBLEM STUFF!! (in the making)
# ===================================================================================


class InverseMonodomainSolverPINNs(MonodomainSolverPINNs):
    """
    Inverse PINN solver for the monodomain equation that estimates the parameter M.
    """
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        device: str,
        source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        initial_M: Union[float, List[float], np.ndarray, torch.Tensor] = 0.1,  # M
        use_ode: bool = False,
        ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_state_vars: int = 0,
        loss_function: str = 'L2',
        loss_weights: Optional[Dict[str, float]] = None
    ):
        # Convert initial_M to tensor if it's a float, list or numpy array
        if isinstance(initial_M, (float, int)):
            initial_M = torch.tensor([initial_M], dtype=torch.float32)
        elif isinstance(initial_M, (list, np.ndarray)):
            initial_M = torch.tensor(initial_M, dtype=torch.float32)

        super(InverseMonodomainSolverPINNs, self).__init__(
            num_inputs=num_inputs,
            num_layers=num_layers,
            num_neurons=num_neurons,
            device=device,
            source_term_func=source_term_func,
            M=initial_M,  # Pass the initial tensor M
            use_ode=use_ode,
            ode_func=ode_func,
            n_state_vars=n_state_vars,
            loss_function=loss_function,
            loss_weights=loss_weights
        )
        
        # Instead of a fixed M, we define M as a trainable parameter with 1 component
        self.M = nn.Parameter(initial_M.to(device))
        self.is_scalar_M = len(initial_M.shape) == 0 or (len(initial_M.shape) == 1 and initial_M.shape[0] == 1)

    def pde(self, X_collocation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the PDE residual loss and ODE residual loss (if applicable).
        """
        # Input is already normalized, no need to scale
        X_collocation.requires_grad_(True)
        
        # Forward pass through the network
        outputs = self.forward(X_collocation)
        
        # Extract u and compute gradients
        if self.use_ode:
            u = outputs[:, 0:1]  # Shape: (N_collocation, 1)
            w = outputs[:, 1:]   # Shape: (N_collocation, n_state_vars)
        else:
            u = outputs  # Shape: (N_collocation, 1)
        
        # Compute gradients
        du_dx = torch.autograd.grad(u.sum(), X_collocation, create_graph=True)[0]
        d2u_dx2 = torch.zeros_like(u)
        
        # Second derivatives for x and y
        d2u_dx2_x = torch.autograd.grad(du_dx[:, 0].sum(), X_collocation, create_graph=True)[0][:, 0:1]
        d2u_dx2_y = torch.autograd.grad(du_dx[:, 1].sum(), X_collocation, create_graph=True)[0][:, 1:2]
        
        # Time derivative
        du_dt = du_dx[:, 2:3]
        
        # Compute Laplacian term
        if self.is_scalar_M:
            laplacian_term = self.M * (d2u_dx2_x + d2u_dx2_y)
        else:
            # For tensor conductivity
            laplacian_term = (
                self.M[0] * d2u_dx2_x +
                self.M[1] * d2u_dx2_y
            )
        
        # Compute source term
        if self.use_ode:
            # Use ODE for current terms
            ode_residuals = self.ode_func(u, w, X_collocation)
            source_term = ode_residuals[:, 0:1]  # First residual corresponds to u
            ode_loss = torch.mean(torch.square(ode_residuals[:, 1:]))  # Remaining residuals
        else:
            # Use explicit source term function
            source_term = self.source_term_func(X_collocation[:, :2], X_collocation[:, 2:3])
            ode_loss = torch.tensor(0.0, device=self.device)
        
        # Compute PDE residual: du/dt - div(M * grad(u)) = I_ion
        pde_residual = du_dt - laplacian_term - source_term
        
        # Compute mean squared PDE residual
        pde_loss = torch.mean(torch.square(pde_residual))
        
        return pde_loss, ode_loss


#======================================================================================
# TESTING SOME NEW STUFF BASED ON THE PAPER "AN EXPERTS GUIDE TO TRAINING PINNS" (S. WANG, S. SANKARAN, H. WANG, P. PERIDKARIS)
#======================================================================================

# Fourier embedding, as it was suggested in the paper

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, num_inputs, mapping_size=256, sigma=5.0):
        """
        Initializes the Fourier Feature Embedding.

        Args:
            num_inputs (int): Dimension of the input features.
            mapping_size (int): Number of random Fourier features (m).
            sigma (float): Scale parameter for the Gaussian distribution (recommended [1, 10]).
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.mapping_size = mapping_size
        self.sigma = sigma

        # Initialize B with shape (mapping_size, num_inputs)
        # Entries are sampled from N(0, sigma^2)
        self.register_buffer(
            'B',
            torch.randn(mapping_size, num_inputs) * sigma
        )

    def forward(self, x):
        """
        Forward pass to compute Fourier embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Fourier embeddings of shape (batch_size, 2 * mapping_size).
        """
        # Compute Bx^T (matrix multiplication)
        # x: (batch_size, num_inputs)
        # B: (mapping_size, num_inputs)
        # Result: (batch_size, mapping_size)
        x_proj = 2 * torch.pi * F.linear(x, self.B)  # Equivalent to x @ B.t()

        # Compute [sin(Bx), cos(Bx)]
        sin_x = torch.sin(x_proj)
        cos_x = torch.cos(x_proj)

        # Concatenate along the feature dimension
        return torch.cat([sin_x, cos_x], dim=-1)  # Shape: (batch_size, 2 * mapping_size)

# Also suggested in the paper
class RandomWeightFactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, mu=1.0, sigma=0.1):
        """
        Initializes the Random Weight Factorized Linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            mu (float): Mean of the normal distribution for initializing scale factors.
            sigma (float): Standard deviation of the normal distribution for initializing scale factors.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize s ~ N(mu, sigma^2)
        self.s = nn.Parameter(torch.randn(out_features) * sigma + mu)  # Shape: (out_features,)

        # Initialize V using Glorot (Xavier) initialization
        self.V = nn.Parameter(torch.empty(out_features, in_features))  # Shape: (out_features, in_features)
        torch.nn.init.xavier_normal_(self.V)

        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        Forward pass for the RWF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Compute diag(exp(s)) * V
        scale_factors = torch.exp(self.s).unsqueeze(1)  # Shape: (out_features, 1)
        scaled_weights = scale_factors * self.V  # Shape: (out_features, in_features)

        # Compute output: x @ W^T + bias
        # W: (out_features, in_features) => W^T: (in_features, out_features)
        output = x @ scaled_weights.t() + self.bias  # Shape: (batch_size, out_features)
        return output



class EnhancedPINN(nn.Module):
    """
    Enhanced PINN class with support for Fourier Feature Embeddings and Random Weight Factorization (RWF).
    """
    def __init__(self, num_inputs, num_layers, num_neurons, num_outputs, device,
                 use_fourier=False, fourier_dim=256, sigma=5.0,
                 use_rwf=False, mu=1.0, sigma_rwf=0.1):
        """
        Initializes the EnhancedPINN.

        Args:
            num_inputs (int): Dimension of the input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons per hidden layer.
            num_outputs (int): Dimension of the output features.
            device (str): Device to run the model on ('cpu' or 'cuda').
            use_fourier (bool): Whether to use Fourier Feature Embeddings.
            fourier_dim (int): Number of random Fourier features (m).
            sigma (float): Scale parameter for Fourier Feature Embeddings.
            use_rwf (bool): Whether to use Random Weight Factorization.
            mu (float): Mean for RWF scale factors.
            sigma_rwf (float): Std dev for RWF scale factors.
        """
        super(EnhancedPINN, self).__init__()
        self.device = device
        self.use_fourier = use_fourier
        self.use_rwf = use_rwf

        if self.use_fourier:
            # Use Fourier feature embeddings with corrected implementation
            self.fourier = FourierFeatureEmbedding(num_inputs, fourier_dim, sigma)
            input_dim = fourier_dim * 2
        else:
            input_dim = num_inputs

        activation = nn.Tanh()
        layers = []

        for layer_idx in range(num_layers + 1):  # Including output layer
            in_dim = input_dim if layer_idx == 0 else num_neurons
            out_dim = num_outputs if layer_idx == num_layers else num_neurons

            if self.use_rwf:
                # Use Random Weight Factorized Linear layers
                layers.append(RandomWeightFactorizedLinear(in_dim, out_dim, mu=mu, sigma=sigma_rwf))
            else:
                # Use standard Linear layers
                linear_layer = nn.Linear(in_dim, out_dim)
                # Initialize weights using Glorot (Xavier) initialization
                torch.nn.init.xavier_normal_(linear_layer.weight)
                torch.nn.init.zeros_(linear_layer.bias)
                layers.append(linear_layer)

            if layer_idx != num_layers:
                # Apply activation after all layers except the last one
                layers.append(activation)

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        """
        Forward pass of the EnhancedPINN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        if self.use_fourier:
            x = self.fourier(x)
        return self.model(x)

class EnhancedMonodomainSolverPINNs(EnhancedPINN):
    """
    Enhanced PINN solver for the monodomain model with support for Fourier embeddings and RWF.
    """
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        device: str,
        source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        M: Union[float, int, list, np.ndarray, torch.Tensor] = None,
        use_ode: bool = False,
        ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_state_vars: int = 0,
        loss_function: str = 'L2',
        loss_weights: Optional[Dict[str, float]] = None,
        # New parameters for Fourier embeddings and RWF
        use_fourier: bool = False,
        fourier_dim: int = 256,
        sigma: float = 5.0,
        use_rwf: bool = False,
        mu: float = 1.0,
        sigma_rwf: float = 0.1,
        M_training: int = 10,  # Number of temporal segments
        epsilon: float = 0.01,  # Hyperparameter for temporal weights
        alpha: float = 0.9,  # Moving average parameter for global weights
        update_frequency: int = 100,  # Frequency of updating global weights
        weight_strategy: str = 'manual',  # 'manual' or 'dynamic'
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0
    ):
        """
        Initializes the EnhancedMonodomainSolverPINNs.

        Args:
            num_inputs (int): Dimension of the input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons per hidden layer.
            device (str): Device to run the model on ('cpu' or 'cuda').
            source_term_func (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): Function defining the source term I_ion.
            M (Union[float, int, list, np.ndarray, torch.Tensor]): Parameter M in the monodomain equation.
            use_ode (bool): Whether to use ODE for current terms.
            ode_func (Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]): Function defining the ODE residuals.
            n_state_vars (int): Number of state variables if using ODE.
            loss_function (str): Type of loss function ('L2' or 'L1').
            loss_weights (Optional[Dict[str, float]]): Weights for different loss components.
            use_fourier (bool): Whether to use Fourier Feature Embeddings.
            fourier_dim (int): Number of random Fourier features (m).
            sigma (float): Scale parameter for Fourier Feature Embeddings.
            use_rwf (bool): Whether to use Random Weight Factorization.
            mu (float): Mean for RWF scale factors.
            sigma_rwf (float): Std dev for RWF scale factors.
        """
        if use_ode:
            output_dimension = 1 + n_state_vars
        else:
            output_dimension = 1

        # Initialize the EnhancedPINN base class
        super(EnhancedMonodomainSolverPINNs, self).__init__(
            num_inputs,
            num_layers,
            num_neurons,
            output_dimension,
            device,
            use_fourier=use_fourier,
            fourier_dim=fourier_dim,
            sigma=sigma,
            use_rwf=use_rwf,
            mu=mu,
            sigma_rwf=sigma_rwf
        )

        # Initialize additional attributes
        self.source_term_func = source_term_func

        if isinstance(M, (float, int, list, np.ndarray)):
            self.M = torch.tensor(M, dtype=torch.float32, device=device)
        elif isinstance(M, torch.Tensor):
            self.M = M.to(device)
        else:
            raise TypeError("M must be a float, int, list, np.ndarray, or torch.Tensor.")

        self.is_scalar_M = self.M.ndim == 0
        self.use_ode = use_ode
        self.ode_func = ode_func
        self.n_state_vars = n_state_vars
        self.loss_function = loss_function

        # Initialize loss weights based on strategy
        self.weight_strategy = weight_strategy
        self.alpha = alpha
        
        if weight_strategy == 'manual':
            if loss_weights is None:
                self.loss_weights = {
                    'pde_loss': 1.0,
                    'IC_loss': 1.0,
                    'BC_loss': 1.0,
                    'data_loss': 1.0,
                    'ode_loss': 1.0
                }
            else:
                self.loss_weights = loss_weights
        else:  # dynamic
            # Initialize dynamic weights
            self.lambda_ic = torch.tensor(1.0, device=device)
            self.lambda_bc = torch.tensor(1.0, device=device)
            self.lambda_r = torch.tensor(1.0, device=device)

        # Initialize data points and expected outputs for data loss
        self.X_data = None
        self.expected_data = None

        # Initialize loss function object for data loss
        if self.loss_function == 'L2':
            self.loss_function_obj = nn.MSELoss()
        elif self.loss_function == 'L1':
            self.loss_function_obj = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def pde(self, X_collocation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the PDE residual loss and ODE residual loss (if applicable).

        Args:
            X_collocation (torch.Tensor): Collocation points tensor of shape (N_collocation, num_inputs).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: PDE loss and ODE loss.
        """
        # Input is already normalized, no need to scale
        X_collocation.requires_grad_(True)
        
        # Forward pass through the network
        outputs = self.forward(X_collocation)
        
        # Extract u and compute gradients
        if self.use_ode:
            u = outputs[:, 0:1]  # Shape: (N_collocation, 1)
            w = outputs[:, 1:]   # Shape: (N_collocation, n_state_vars)
        else:
            u = outputs  # Shape: (N_collocation, 1)
        
        # Compute gradients
        du_dx = torch.autograd.grad(u.sum(), X_collocation, create_graph=True)[0]
        d2u_dx2 = torch.zeros_like(u)
        
        # Second derivatives for x and y
        d2u_dx2_x = torch.autograd.grad(du_dx[:, 0].sum(), X_collocation, create_graph=True)[0][:, 0:1]
        d2u_dx2_y = torch.autograd.grad(du_dx[:, 1].sum(), X_collocation, create_graph=True)[0][:, 1:2]
        
        # Time derivative
        du_dt = du_dx[:, 2:3]
        
        # Compute Laplacian term
        if self.is_scalar_M:
            laplacian_term = self.M * (d2u_dx2_x + d2u_dx2_y)
        else:
            # For tensor conductivity
            laplacian_term = (
                self.M[0] * d2u_dx2_x +
                self.M[1] * d2u_dx2_y
            )
        
        # Compute source term
        if self.use_ode:
            # Use ODE for current terms
            ode_residuals = self.ode_func(u, w, X_collocation)
            source_term = ode_residuals[:, 0:1]  # First residual corresponds to u
            ode_loss = torch.mean(torch.square(ode_residuals[:, 1:]))  # Remaining residuals
        else:
            # Use explicit source term function
            source_term = self.source_term_func(X_collocation[:, :2], X_collocation[:, 2:3])
            ode_loss = torch.tensor(0.0, device=self.device)
        
        # Compute PDE residual: du/dt - div(M * grad(u)) = I_ion
        pde_residual = du_dt - laplacian_term - source_term
        
        # Compute mean squared PDE residual
        pde_loss = torch.mean(torch.square(pde_residual))
        
        return pde_loss, ode_loss

    def IC(self, X_ic: torch.Tensor, expected_u0: torch.Tensor) -> torch.Tensor:
        """
        Compute the Initial Condition (IC) loss.

        Args:
            X_ic (torch.Tensor): Initial condition points tensor of shape (N_IC, num_inputs).
            expected_u0 (torch.Tensor): Expected initial values tensor of shape (N_IC, 1).

        Returns:
            torch.Tensor: IC loss.
        """
        # The input X_ic is already normalized, no need to scale it
        X_ic.requires_grad_(True)
        
        # Forward pass through the network with scaled input
        outputs = self.forward(X_ic)
        
        # Extract u based on whether ODE is used
        if self.use_ode:
            u0 = outputs[:, 0:1]  # Assuming u0 is the first output
        else:
            u0 = outputs  # Shape: (N, 1)
        
        # Compute loss using unscaled values
        if self.loss_function == 'L2':
            IC_loss = nn.MSELoss()(u0, expected_u0.to(self.device))
        elif self.loss_function == 'L1':
            IC_loss = nn.L1Loss()(u0, expected_u0.to(self.device))
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        return IC_loss

    def BC_neumann(
        self,
        X_boundary: torch.Tensor,
        normal_vectors: torch.Tensor,
        expected_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Neumann Boundary Condition (BC) loss.

        Args:
            X_boundary (torch.Tensor): Boundary points tensor of shape (N_BC, num_inputs).
            normal_vectors (torch.Tensor): Normal vectors at boundary points of shape (N_BC, num_spatial_dims).
            expected_value (Optional[torch.Tensor]): Expected Neumann flux values, default is zero.

        Returns:
            torch.Tensor: BC loss.
        """
        # Input is already normalized, no need to scale
        X_boundary.requires_grad_(True)
        
        # Forward pass through the network
        outputs = self.forward(X_boundary)
        
        # Extract u
        if self.use_ode:
            u = outputs[:, 0:1]  # Shape: (N_BC, 1)
        else:
            u = outputs  # Shape: (N_BC, 1)
        
        # Compute spatial derivatives
        du_dx = torch.autograd.grad(u.sum(), X_boundary, create_graph=True)[0][:, :2]
        
        # Compute normal derivative
        if self.is_scalar_M:
            normal_deriv = self.M * torch.sum(du_dx * normal_vectors, dim=1, keepdim=True)
        else:
            # For tensor conductivity
            normal_deriv = torch.sum(
                torch.stack([
                    self.M[0] * du_dx[:, 0],
                    self.M[1] * du_dx[:, 1]
                ]) * normal_vectors.t(),
                dim=0,
                keepdim=True
            ).t()
        
        # If no expected value is provided, assume zero Neumann BC
        if expected_value is None:
            expected_value = torch.zeros_like(normal_deriv)
        
        # Compute BC loss
        bc_loss = torch.mean(torch.square(normal_deriv - expected_value))
        
        return bc_loss

    def data_loss(self, X_data: torch.Tensor, expected_data: torch.Tensor) -> torch.Tensor:
        """
        Compute the data loss between the model's predictions and expected data.

        Args:
            X_data (torch.Tensor): Data points tensor of shape (N_data, num_inputs).
            expected_data (torch.Tensor): Expected data tensor of shape (N_data, num_outputs).

        Returns:
            torch.Tensor: Data loss.
        """
        # Forward pass through the network
        outputs = self.forward(X_data)

        # Extract u based on whether ODE is used
        if self.use_ode:
            u = outputs[:, 0:1]  # Assuming u is the first output
        else:
            u = outputs  # Shape: (N_data, num_outputs)

        # Compute loss
        if self.loss_function in ['L2', 'L1']:
            loss = self.loss_function_obj(u, expected_data.to(self.device))
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

        return loss

    def train_step(self, optimizer: torch.optim.Optimizer, batch_size: Optional[int] = None) -> tuple:
        """
        Perform a single training step.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            batch_size (Optional[int]): Size of each mini-batch. If None, use full dataset.

        Returns:
            tuple: Tuple containing individual loss components and total loss.
        """
        self.train()
        optimizer.zero_grad()

        # Get mini-batches if batch_size is specified
        if batch_size is not None:
            # Generate separate indices for each dataset since they might have different sizes
            collocation_indices = torch.randperm(self.X_collocation.size(0))[:batch_size]
            ic_indices = torch.randperm(self.X_ic.size(0))[:batch_size]
            boundary_indices = torch.randperm(self.X_boundary.size(0))[:batch_size]

            X_collocation_batch = self.X_collocation[collocation_indices]
            X_ic_batch = self.X_ic[ic_indices]
            expected_u0_batch = self.expected_u0[ic_indices]
            X_boundary_batch = self.X_boundary[boundary_indices]
            normal_vectors_batch = self.normal_vectors[boundary_indices]

            if self.X_data is not None:
                data_indices = torch.randperm(self.X_data.size(0))[:batch_size]
                X_data_batch = self.X_data[data_indices]
                expected_data_batch = self.expected_data[data_indices]
            else:
                X_data_batch = None
                expected_data_batch = None
        else:
            # Use entire dataset
            X_collocation_batch = self.X_collocation
            X_ic_batch = self.X_ic
            expected_u0_batch = self.expected_u0
            X_boundary_batch = self.X_boundary
            normal_vectors_batch = self.normal_vectors
            X_data_batch = self.X_data
            expected_data_batch = self.expected_data

        # Update weights if using dynamic strategy
        if self.weight_strategy == 'dynamic':
            self.update_weights(
                X_collocation_batch,
                X_ic_batch,
                expected_u0_batch,
                X_boundary_batch,
                normal_vectors_batch
            )

        # Compute losses
        IC_loss = self.IC(X_ic_batch, expected_u0_batch)
        BC_loss = self.BC_neumann(X_boundary_batch, normal_vectors_batch)
        pde_loss, ode_loss = self.pde(X_collocation_batch)

        # Apply loss weights and aggregate
        if self.weight_strategy == 'manual':
            total_loss = (
                self.loss_weights['IC_loss'] * IC_loss +
                self.loss_weights['BC_loss'] * BC_loss +
                self.loss_weights['pde_loss'] * pde_loss +
                self.loss_weights['ode_loss'] * ode_loss
            )
        else:  # dynamic
            total_loss = (
                self.lambda_ic * IC_loss +
                self.lambda_bc * BC_loss +
                self.lambda_r * pde_loss +
                ode_loss
            )

        # Compute data loss if applicable
        if X_data_batch is not None and expected_data_batch is not None:
            data_loss = self.data_loss(X_data_batch, expected_data_batch)
            if self.weight_strategy == 'manual':
                total_loss += self.loss_weights['data_loss'] * data_loss
            else:  # dynamic
                total_loss += data_loss
        else:
            data_loss = torch.tensor(0.0, device=self.device)

        # Backpropagation with retain_graph=True
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # Return loss values
        return pde_loss.item(), IC_loss.item(), BC_loss.item(), data_loss.item(), ode_loss.item(), total_loss.item()

    def evaluate(self, X_test: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple]:
        """
        Evaluate the trained model on test data.

        Args:
            X_test (torch.Tensor): Test points tensor of shape (N_test, num_inputs).
            y_true (Optional[torch.Tensor]): True values tensor of shape (N_test, num_outputs).

        Returns:
            Union[torch.Tensor, tuple]: If y_true is provided, returns (predictions, loss). Otherwise, returns predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_test.to(self.device))

        # Extract u based on whether ODE is used
        if self.use_ode:
            y_pred = outputs[:, 0:1]  # Only u
        else:
            y_pred = outputs  # Shape: (N_test, num_outputs)

        if y_true is not None:
            if self.loss_function in ['L2', 'L1']:
                loss = self.loss_function_obj(y_pred, y_true.to(self.device)).item()
            else:
                raise ValueError(f"Unknown loss function: {self.loss_function}")
            return y_pred.cpu(), loss
        else:
            return y_pred.cpu()

    def validate(self, X_collocation_val: torch.Tensor, X_ic_val: torch.Tensor, expected_u0_val: torch.Tensor, 
                 X_boundary_val: torch.Tensor, normal_vectors_val: torch.Tensor) -> float:
        """
        Validate the model on validation data.

        Args:
            X_collocation_val (torch.Tensor): Validation collocation points tensor.
            X_ic_val (torch.Tensor): Validation initial condition points tensor.
            expected_u0_val (torch.Tensor): Expected initial values tensor.
            X_boundary_val (torch.Tensor): Validation boundary points tensor.
            normal_vectors_val (torch.Tensor): Normal vectors at boundary points.

        Returns:
            float: Total validation loss.
        """
        self.eval()
        with torch.enable_grad():  # Enable gradients only within this block
            # Compute PDE Loss
            pde_loss_val, ode_loss_val = self.pde(X_collocation_val.to(self.device))

            # Compute IC Loss
            ic_loss_val = self.IC(X_ic_val.to(self.device), expected_u0_val.to(self.device))

            # Compute BC Loss
            bc_loss_val = self.BC_neumann(
                X_boundary_val.to(self.device), 
                normal_vectors_val.to(self.device)
            )

            # Compute Data Loss if applicable
            if self.X_data is not None and self.expected_data is not None:
                data_loss_val = self.data_loss(
                    self.X_data.to(self.device), 
                    self.expected_data.to(self.device)
                )
            else:
                data_loss_val = torch.tensor(0.0, device=self.device)

            # Aggregate Total Validation Loss
            if self.weight_strategy == 'manual':
                total_val_loss = (
                    self.loss_weights['pde_loss'] * pde_loss_val +
                    self.loss_weights['IC_loss'] * ic_loss_val +
                    self.loss_weights['BC_loss'] * bc_loss_val +
                    self.loss_weights['data_loss'] * data_loss_val +
                    self.loss_weights['ode_loss'] * ode_loss_val
                )
            else:  # dynamic
                total_val_loss = (
                    self.lambda_ic * ic_loss_val +
                    self.lambda_bc * bc_loss_val +
                    self.lambda_r * pde_loss_val +
                    ode_loss_val +
                    data_loss_val
                )

        return total_val_loss.item()

    def save_model(self, file_path: str):
        """
        Save the trained model's state dictionary to a file.

        Args:
            file_path (str): Path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """
        Load the model's state dictionary from a file.

        Args:
            file_path (str): Path to the file from which the model will be loaded.
        """
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.to(self.device)
        self.eval()
        print(f"Model loaded from {file_path}")

    def compute_gradient_norms(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        """Compute L2 norms of gradients for each loss component."""
        self.zero_grad()
        
        # Compute losses
        pde_loss, _ = self.pde(X_collocation)
        ic_loss = self.IC(X_ic, expected_u0)
        bc_loss = self.BC_neumann(X_boundary, normal_vectors)
        
        # Compute gradients with allow_unused=True and retain_graph=True
        grad_r = torch.autograd.grad(pde_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_ic = torch.autograd.grad(ic_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_bc = torch.autograd.grad(bc_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        
        # Filter out None gradients and compute L2 norms, then detach
        norm_r = torch.sqrt(sum(torch.sum(g**2) for g in grad_r if g is not None)).detach()
        norm_ic = torch.sqrt(sum(torch.sum(g**2) for g in grad_ic if g is not None)).detach()
        norm_bc = torch.sqrt(sum(torch.sum(g**2) for g in grad_bc if g is not None)).detach()
        
        return norm_r, norm_ic, norm_bc

    def update_weights(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        """Update weights using gradient norms and moving average."""
        if self.weight_strategy != 'dynamic':
            return
            
        # Compute gradient norms
        norm_r, norm_ic, norm_bc = self.compute_gradient_norms(
            X_collocation, X_ic, expected_u0, X_boundary, normal_vectors
        )
        
        # Compute sum of norms
        sum_norms = (norm_r + norm_ic + norm_bc).detach()
        
        # Compute new weights
        lambda_ic_new = (sum_norms / (norm_ic + 1e-8)).detach()  # Add small epsilon to prevent division by zero
        lambda_bc_new = (sum_norms / (norm_bc + 1e-8)).detach()
        lambda_r_new = (sum_norms / (norm_r + 1e-8)).detach()
        
        # Update weights using moving average
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * lambda_ic_new
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * lambda_bc_new
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * lambda_r_new

'''
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
        X_collocation.requires_grad_(True)
        x = X_collocation[:, 0:1]
        y = X_collocation[:, 1:2]
        t = X_collocation[:, 2:3]

        outputs = self.model(torch.cat([x, y, t], dim=1))
        v = outputs[:, 0:1]
        ue = outputs[:, 1:2]

        # First derivatives
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]


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
'''