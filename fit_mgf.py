import os
import math
import mpmath
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

class HolomorphicLinearOld(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False):
        super().__init__()
        denom = max(in_features, 1)
        scale = (1 / denom) if is_first else (np.sqrt(6) / (omega_0 * np.sqrt(denom))) #0.01
        scale *= 0.01
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.cdouble).uniform_(-scale, scale)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, dtype=torch.cdouble).uniform_(-scale, scale)
        ) #nn.Parameter(torch.zeros(out_features, dtype=torch.cdouble))

    def forward(self, z):
        return torch.nn.functional.linear(z, self.weight, self.bias)

class HolomorphicLinear(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False):
        super().__init__()

        denom = max(in_features, 1)
        scale = (1 / denom) if is_first else (np.sqrt(6) / (omega_0 * np.sqrt(denom)))
        scale *= 0.01

        # Real and imaginary parts of weight
        self.Wr = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-scale, scale)
        )
        self.Wi = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-scale, scale)
        )

        # Real and imaginary parts of bias
        self.br = nn.Parameter(
            torch.empty(out_features).uniform_(-scale, scale)
        )
        self.bi = nn.Parameter(
            torch.empty(out_features).uniform_(-scale, scale)
        )

    def forward(self, z):
        """
        z: complex tensor of shape (..., in_features)
        """
        zr = z.real
        zi = z.imag

        real = zr @ self.Wr.T - zi @ self.Wi.T + self.br
        imag = zr @ self.Wi.T + zi @ self.Wr.T + self.bi
#        real = zr @ self.Wr.T + self.br
#        imag = zr @ self.Wi.T + self.bi

        return torch.complex(real, imag)

class NormalizeComplex(nn.Module):
    def __init__(self, max_mag=3.0, eps=1e-8):
        super().__init__()
        self.max_mag = max_mag
        self.eps = eps

    def forward(self, z):
        mag = torch.abs(z)
        scale = torch.clamp(self.max_mag / (mag + self.eps), max=1.0)
        return z * scale

class ComplexExpGate(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.double))

    def forward(self, z):
        return z * torch.exp(self.alpha * z)

class PolyResidualBlock(nn.Module):
    """
    Holomorphic multivariate polynomial residual block:
        z -> z + A z + sum_{i,j} B_{ij} z_i z_j
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        scale = 0.1
        # Linear perturbation
        self.A = nn.Parameter(
            torch.randn(d, d, dtype=torch.cdouble) * scale
        )
        # Quadratic cross terms
        self.B = nn.Parameter(
            torch.randn(d, d, d, dtype=torch.cdouble) * scale
        )

    def forward(self, z):
        # z: (batch, d)
        linear = z @ self.A.T                      # (batch, d)
        # quadratic: sum_jk B[i,j,k] z_j z_k
        quad = torch.einsum("bij,jk->bi", self.B, torch.einsum("bj,bk->jk", z, z))
        return z + linear + quad

class CauchyActivation(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        return 1.0 / (1.0 + z * z + self.eps)

class BoundedHolomorphicActivation(nn.Module):
    def forward(self, z):
        # entire + bounded in imaginary direction
        return z / (1 + z*z)

class NormalizeActivation(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        mag = torch.abs(z)
        return z / (1.0 + mag + self.eps)

class ComplexSine(nn.Module):
    def __init__(self, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, z):
        return torch.sin(self.omega_0 * z)

class FourierFeatures(nn.Module):
    def __init__(self, num_features=32):
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError("num_features must be even")
        self.num_features = num_features
        half = num_features // 2

        freqs_x = torch.logspace(0, 1, half, dtype=torch.float64)
        freqs_y = torch.logspace(0, 2, half, dtype=torch.float64)

        # Shape: (1, 1, half) for broadcasting
        self.register_buffer("freqs_x", freqs_x.view(1, 1, -1))
        self.register_buffer("freqs_y", freqs_y.view(1, 1, -1))

    def forward(self, x):
        """
        x: complex tensor of shape (N, d)
        returns: real tensor of shape (N, d * num_features)
        """
        x_real = x.real.unsqueeze(-1)  # (N, d, 1)
        x_imag = x.imag.unsqueeze(-1)  # (N, d, 1)

        xb = 2.0 * math.pi * x_real * self.freqs_x  # (N, d, half)
        yb = 2.0 * math.pi * x_imag * self.freqs_y  # (N, d, half)

        feats = torch.cat(
            [
                torch.sin(xb),
                torch.cos(xb),
                torch.sin(yb),
                torch.cos(yb),
            ],
            dim=-1,  # (N, d, num_features)
        )

        return feats.view(x.shape[0], -1)  # (N, d * num_features)

class LogGMFNet(nn.Module):
    def __init__(self, d, ff_m = 32, hidden_dim = 128, scale_by_zero = False, x_min = -1, x_max = 0, y_min = -1, y_max = 1):
        super().__init__()
        self.d = d
        self.ff = FourierFeatures(ff_m)
        self.net = nn.Sequential(
            nn.Linear(self.d * ff_m * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # (Re log g, Im log g)
        )
        self.X_MIN, self.X_MAX = x_min, x_max
        self.Y_MIN, self.Y_MAX = y_min, y_max
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        x_coord = x.real
        y_coord = x.imag

        x_coord = 2.0 * (x_coord - self.X_MIN) / (self.X_MAX - self.X_MIN) - 1.0
        y_coord = 2.0 * (y_coord - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) - 1.0
        x_norm = torch.complex(x_coord, y_coord) #torch.cat([x_coord, y_coord], dim=1)
        
        raw = self.net(self.ff(x_norm))
        raw = torch.complex(raw[:,0:1], raw[:,1:2])
        if self.scale_by_zero:
            x_zero = torch.zeros_like(x_coord, dtype=torch.double)
            y_zero = torch.zeros_like(y_coord, dtype=torch.double)
            x_zero = 2.0 * (x_zero - self.X_MIN) / (self.X_MAX - self.X_MIN) - 1.0
            y_zero = 2.0 * (y_zero - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) - 1.0
            zero_point = torch.complex(x_zero, y_zero) #torch.cat([x_zero, y_zero], dim=1) #torch.zeros(1, 2, dtype = torch.double, device = x.device)
            raw0 = self.net(self.ff(zero_point))
            raw0 = torch.complex(raw0[:,0:1], raw0[:,1:2])
            output = raw - raw0
        else:
            output = raw
        return output

class FFNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, omega_0 = 5.0, scale_by_zero = False):
        super(FFNet, self).__init__()
        self.network = nn.Sequential(
            HolomorphicLinear(input_dim, hidden_dim, omega_0, is_first = True),
            nn.Tanh(),
            HolomorphicLinear(hidden_dim, 64, omega_0),
            nn.Tanh(),
            HolomorphicLinear(64, 64, omega_0),
            nn.Tanh(),
            HolomorphicLinear(64, output_dim, omega_0),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.network(x)
        # Combine into a complex-valued "raw" network output
        if self.scale_by_zero:
            zero_point = torch.zeros(1, x.shape[1], dtype=torch.cdouble, device = raw.device)
            raw0 = self.network(zero_point)
            output = torch.exp(raw - raw0)
        else:
            output = torch.exp(raw)
        return output

class RealFFNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, scale_by_zero = False):
        super(FFNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, is_first = True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.network(x)
        # Combine into a complex-valued "raw" network output
        if self.scale_by_zero:
            zero_point = torch.zeros(1, x.shape[1], dtype=torch.cdouble, device = raw.device)
            raw0 = self.network(zero_point)
            output = torch.exp(raw - raw0)
        else:
            output = torch.exp(raw)
        return output

class PolyResNet(nn.Module):
    """
    Multivariate holomorphic polynomial residual network
    with optional scale_by_zero normalization.
    """
    def __init__(self, input_dim, depth=4, scale_by_zero=False):
        super().__init__()
        self.input_dim = input_dim
        self.scale_by_zero = scale_by_zero

        self.blocks = nn.ModuleList(
            [PolyResidualBlock(input_dim) for _ in range(depth)]
        )

        # Final holomorphic projection to scalar log-MGF
        self.c = nn.Parameter(
            torch.randn(input_dim, 1, dtype=torch.cdouble) * 0.1
        )

    def core(self, theta):
        """
        Core holomorphic map producing log-MGF (unnormalized).
        """
        z = theta
        for block in self.blocks:
            z = block(z)
        return z @ self.c   # (batch, 1)

    def forward(self, theta):
        """
        Returns MGF(theta), not log-MGF.
        """
        if self.input_dim == 0:
            return torch.ones(theta.shape[0], 1, dtype=torch.cdouble)
            
        raw = self.core(theta)

        if self.scale_by_zero:
            zero_point = torch.zeros(
                1, self.input_dim, dtype=torch.cdouble, device=theta.device
            )
            raw0 = self.core(zero_point)
            output = torch.exp(raw - raw0)
        else:
            output = torch.exp(raw)

        return output

class BoundedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, omega_0 = 5.0, scale_by_zero = False):
        super(BoundedNet, self).__init__()
        self.network = nn.Sequential(
            HolomorphicLinear(input_dim, hidden_dim, omega_0, is_first = True),
            BoundedHolomorphicActivation(),
            HolomorphicLinear(hidden_dim, 64, omega_0),
            BoundedHolomorphicActivation(),
            HolomorphicLinear(64, output_dim, omega_0),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.network(x)
        # Combine into a complex-valued "raw" network output
        if self.scale_by_zero:
            zero_point = torch.zeros(1, x.shape[1], dtype=torch.cdouble, device = raw.device)
            raw0 = self.network(zero_point)
            output = torch.exp(raw - raw0)
        else:
            output = torch.exp(raw)
        return output

class SirenNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, omega_0 = 5.0, scale_by_zero = False):
        super(SirenNet, self).__init__()
        self.C = 3
        self.hf_net = nn.Sequential(
            HolomorphicLinear(input_dim, hidden_dim, omega_0, is_first = True),
            NormalizeComplex(3.0),
            ComplexSine(omega_0),
            HolomorphicLinear(hidden_dim, 64, omega_0, is_first = False),
            NormalizeComplex(3.0),
            ComplexSine(omega_0),
            HolomorphicLinear(64, output_dim, omega_0),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.hf_net(x)
        if self.scale_by_zero:
            zero = torch.zeros(1, x.shape[1], dtype=torch.cdouble, device=x.device)
            raw0 = self.hf_net(zero)
            return torch.exp(raw - raw0)
        return torch.exp(raw)

class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, omega_0 = 5.0, scale_by_zero = False):
        super(LinearNet, self).__init__()
        self.network = nn.Sequential(
            HolomorphicLinear(input_dim, hidden_dim, omega_0, is_first = True),
            HolomorphicLinear(hidden_dim, output_dim, omega_0),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.network(x)
        # Combine into a complex-valued "raw" network output
        if self.scale_by_zero:
            zero_point = torch.zeros(1, x.shape[1], dtype=torch.cdouble, device = raw.device)
            raw0 = self.network(zero_point)
            output = torch.exp(raw - raw0)
        else:
            output = torch.exp(raw)
        return output
    
class MGFNet(nn.Module):
    def __init__(self, d, hidden_dim = 64, x_min = -3, x_max = -0.5, y_min = -16, y_max = 0):
        super(MGFNet, self).__init__()
        self.d = d
        omega_0 = 1.0
#        self.interior_network = FFNet(self.d, 1, hidden_dim = hidden_dim, omega_0 = omega_0, scale_by_zero = True)
        self.interior_network = LogGMFNet(self.d, ff_m = 32, hidden_dim = hidden_dim, scale_by_zero = True, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max) #PolyResNet(self.d, depth = 1, scale_by_zero = True)
        self.boundary_networks = nn.ModuleList()
        for i in range(self.d):
            self.boundary_networks.append(LogGMFNet(self.d - 1, ff_m = 32, hidden_dim = hidden_dim, scale_by_zero = False, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max))
#            self.boundary_networks.append(FFNet(self.d-1, 1, hidden_dim = hidden_dim, omega_0 = omega_0))

    def forward(self, x):
        phi = self.interior_network(x)
        phi_i = torch.zeros((x.shape[0], self.d), dtype=torch.cdouble, device = phi.device)
        for i in range(self.d):
            input_i = torch.concat([x[:,:i], x[:,(i+1):]], dim = 1)
            phi_i[:,i] = self.boundary_networks[i](input_i).flatten()
        return torch.concat([phi, phi_i], dim = 1)
    
    def freeze_all(self):
        self.freeze_interior()
        self.freeze_boundary()
    
    def unfreeze_all(self):
        self.unfreeze_interior()
        self.unfreeze_boundary()
    
    def freeze_interior(self):
        for param in self.interior_network.parameters():
            param.requires_grad = False
    
    def unfreeze_interior(self):
        for param in self.interior_network.parameters():
            param.requires_grad = True
    
    def freeze_boundary(self):
        for param in self.boundary_networks.parameters():
            param.requires_grad = False
    
    def unfreeze_boundary(self):
        for param in self.boundary_networks.parameters():
            param.requires_grad = True
    
    def freeze_boundary_i(self, i):
        for param in self.boundary_networks[i].parameters():
            param.requires_grad = False
    
    def unfreeze_boundary_i(self, i):
        for param in self.boundary_networks[i].parameters():
            param.requires_grad = True

class MGFTrainer:
    def __init__(self, d, mu, sigma, R, hidden_dim = 128, dir = ".", x_min = -3, x_max = -0.5, y_min = -16, y_max = 0):
        self.d = d
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if mu is not None and sigma is not None and R is not None:
            self.MU = torch.complex(mu, torch.zeros_like(mu)).to(device = self.device)
            self.SIGMA = torch.complex(sigma, torch.zeros_like(sigma)).to(device = self.device)
            self.R = torch.complex(R, torch.zeros_like(R)).to(device = self.device)
        self.model = MGFNet(self.d, hidden_dim = hidden_dim).double().to(device = self.device)
        self.dir = dir
        self.engine = SobolEngine(dimension=d)
    
    # ---- Define monotonicity penalty ----
    def monotonicity_penalty(self, model, s):
        """
        Penalizes negative slopes of the real part of a complex-valued model output.
        """
        s_zero_imag = torch.complex(s.real, torch.zeros_like(s.imag).double())
        s_zero_imag.requires_grad_(True)
        M_pred = model(s_zero_imag)  # complex output, shape [N, ...]
        
        # Take the real part for monotonicity
        M_real = M_pred.real
        
        # Compute gradients w.r.t input
        grad_real = torch.autograd.grad(M_real.sum(), s_zero_imag, create_graph=True)[0]
        
        # Penalize negative slopes
        penalty = torch.relu(-grad_real.real).mean() + 0.1 * torch.mean(torch.abs(M_pred.imag) ** 2)
        return penalty
    
    def cauchy_riemann_penalty(self, model, z):
        """
        Enforces Cauchy–Riemann equations by differentiating with respect
        to real coordinates (x, y).

        z: complex tensor of shape (N, d) or (N, 1)
        model(z) -> complex output
        """

        # Split complex input into real variables
        x = z.real.clone().detach().requires_grad_(True)
        y = z.imag.clone().detach().requires_grad_(True)

        z_xy = torch.complex(x, y)
        fz = model(z_xy)

        u = fz.real
        v = fz.imag

        # Gradients w.r.t. real variables
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]

        # Cauchy–Riemann residual
        cr_penalty = torch.mean((u_x - v_y)**2 + (u_y + v_x)**2)
        return cr_penalty
    
    def growth_penalty(self, model, s, C=1.0):
        M = model(s)
        bound = torch.exp(-C * torch.norm(s, dim=1))
        bound = bound.unsqueeze(1)
        return torch.mean(torch.relu(torch.abs(M) - bound)**2)

    def sample_vector(self, lb=-1, ub=0, imag_lb=-0.5, imag_ub=0.5, batch_size=100):
        # Draw real part
#        real_part = (ub - lb) * self.engine.draw(batch_size) + lb
#        real_part = real_part.double().to(device = self.device)
#
#        # Draw imaginary part independently
#        imag_part = (imag_ub - imag_lb) * self.engine.draw(batch_size) + imag_lb
#        imag_part = imag_part.double().to(device = self.device)
        real_part = (ub - lb) * torch.rand(batch_size, self.d, device = self.device) + lb
        real_part = real_part.double().to(device = self.device)

        # Draw imaginary part independently
        imag_part = (imag_ub - imag_lb) * torch.rand(batch_size, self.d, device = self.device) + imag_lb
        imag_part = imag_part.double().to(device = self.device)

        # Combine into complex tensor
        vec = torch.complex(real_part, imag_part)
        return vec
    
    ## Assume theta is a N x d matrix
    def gamma(self, theta):
        gamma_theta = -(0.5 * torch.diagonal(theta @ self.SIGMA @ theta.T) + (theta @ self.MU).flatten()).flatten()
        gamma_i_theta = theta @ self.R
        return gamma_theta, gamma_i_theta

    ## Phi_i_theta: N x d
    def bar_loss(self, theta, phi_theta, phi_i_theta):
        gamma_theta, gamma_i_theta = self.gamma(theta)
        lhs = gamma_theta * phi_theta
        rhs = torch.sum(gamma_i_theta * phi_i_theta, dim = 1)
        diff = (lhs - rhs)
        return torch.mean(torch.abs(diff) ** 2)
    
    def log_bar_loss(self, theta, log_phi_theta, log_phi_i_theta, train_idx = 0):
        gamma_theta, gamma_i_theta = self.gamma(theta)
        if train_idx == 0:
            lhs = torch.log(gamma_theta) + log_phi_theta
            rhs = torch.log(torch.sum(gamma_i_theta * torch.exp(log_phi_i_theta), dim = 1))
        else:
            lhs = torch.log(gamma_i_theta[:,train_idx-1]) + log_phi_i_theta[:,train_idx-1]
            lhs_left = torch.sum(gamma_i_theta[:,:max(train_idx-1, 0)] * torch.exp(log_phi_i_theta[:,:max(train_idx-1, 0)]), dim = 1)
            lhs_right = torch.sum(gamma_i_theta[:,train_idx:] * torch.exp(log_phi_i_theta[:,train_idx:]), dim = 1)
            rhs = torch.log(gamma_theta * torch.exp(log_phi_theta) - lhs_left - lhs_right)
        diff = (lhs - rhs)
        return torch.mean(torch.abs(diff) ** 2)
    
#    def log_bar_loss(self, theta, log_phi, log_phi_i):
#        """
#        Overflow-safe BAR loss using max-normalization.
#
#        Returns: scalar real loss
#        """
#
#        gamma_theta, gamma_i_theta = self.gamma(theta)
#        eps = 1e-30
#        # ----------------------------
#        # Step 1: log-magnitudes a_j
#        # ----------------------------
#        # a0 = log |gamma * exp(log_phi)|
#        a0 = torch.log(torch.abs(gamma_theta) + eps) + log_phi.real.squeeze(-1)  # (N,)
#        # ai = log |gamma_i * exp(log_phi_i)|
#        ai = torch.log(torch.abs(gamma_i_theta) + eps) + log_phi_i.real           # (N, d)
#        # Stack for max
#        a_all = torch.cat([a0.unsqueeze(1), ai], dim=1)  # (N, d+1)
#        m = torch.max(a_all, dim=1, keepdim=True)[0]     # (N, 1)
#
#        # ----------------------------
#        # Step 2: phases
#        # ----------------------------
#        phi0 = torch.angle(gamma_theta) + log_phi.imag.squeeze(-1)  # (N,)
#        phii = torch.angle(gamma_i_theta) + log_phi_i.imag          # (N, d)
#
#        # ----------------------------
#        # Step 3: scaled complex terms
#        # ----------------------------
#        t0 = torch.exp(a0.unsqueeze(1) - m) * torch.exp(1j * phi0.unsqueeze(1))  # (N, 1)
#        ti = torch.exp(ai - m) * torch.exp(1j * phii)                             # (N, d)
#
#        # ----------------------------
#        # Step 4: BAR residual
#        # ----------------------------
#        residual = t0 - torch.sum(ti, dim=1, keepdim=True)  # (N, 1)
#
#        # ----------------------------
#        # Step 5: squared magnitude loss
#        # ----------------------------
#        loss = torch.mean(torch.abs(residual) ** 2)
#        return loss

    
    def train_from_target(self, target_mgf_func, full_gradient = False, theta_eval = None, lb = -1, ub = 0, imag_lb=-0.5, imag_ub=0.5, batch_size = 500, num_epochs = 10000, init_lr = 1e-3, lam_monotone = 0.1, lam_CR = 1e-3, lam_growth = 1e-4):
        if full_gradient:
            assert theta_eval is not None
        optimizer = optim.AdamW(self.model.parameters(), lr = init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=init_lr * 0.02)
#        scheduler = ExponentialLR(optimizer, gamma=0.99)
#        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#            optimizer,
#            T_0=100,       # number of steps before first restart
#            T_mult=1,     # how much T increases after restart
#            eta_min=1e-6  # minimum LR
#        )
        loss_arr = []
        log_loss_arr = []
        loss_rel_arr = []
        loss_cr_arr = []
        for epoch in tqdm(range(num_epochs)):
            if full_gradient:
                theta = theta_eval.clone()
                theta = theta.to(device = self.device)
            else:
                theta = self.sample_vector(lb = lb, ub = ub, imag_lb=imag_lb, imag_ub=imag_ub, batch_size = batch_size)
            # if theta_eval is not None:
            #     anchors = theta_eval.clone()
            #     N = anchors.shape[0]
            #     idx = torch.randint(0, N, size=(batch_size,), device=anchors.device)
            #     anchors = anchors[idx]
            #     anchors = anchors.to(device = self.device)
            #     theta = torch.cat([theta, anchors], dim = 0)
            output = self.model(theta)
            log_phi_theta = output[:,0].view((-1, 1))
            phi_theta = torch.exp(log_phi_theta)
            log_phi_i_theta = output[:,1:].view((-1, self.d))
            phi_i_theta = torch.exp(log_phi_i_theta)
            phi_theta_true = target_mgf_func(theta)
            loss = torch.mean(torch.abs(log_phi_theta - torch.log(phi_theta_true)) ** 2)
            log_loss_arr.append(loss.item())
            loss_rel = torch.mean(torch.abs(phi_theta - phi_theta_true) / torch.abs(phi_theta_true))
            loss_rel_arr.append(loss_rel.item())
#            loss = torch.mean(torch.abs(phi_theta - phi_theta_true) ** 2)
            if lam_monotone > 0:
                loss += lam_monotone * self.monotonicity_penalty(self.model, theta)
            if lam_CR > 0:
                loss_cr = self.cauchy_riemann_penalty(self.model, theta)
                loss_cr_arr.append(loss_cr.item())
                loss += lam_CR * loss_cr
            if lam_growth > 0:
                loss += lam_growth * self.growth_penalty(self.model, theta)
            if torch.isnan(loss):
                print("NaN produced in training.")
                assert False
            loss_arr.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        ## Evaluation
        plt.plot(loss_arr)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.yscale("log")
        plt.title(f"{loss_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/loss.png")
        plt.clf()
        plt.close()
        
        plt.plot(log_loss_arr)
        plt.xlabel("Epoch")
        plt.ylabel("Log MSE")
        #plt.yscale("log")
        plt.title(f"{log_loss_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/log_loss.png")
        plt.clf()
        plt.close()
        
        plt.plot(loss_rel_arr)
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        #plt.yscale("log")
        plt.title(f"{loss_rel_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/rel_loss.png")
        plt.clf()
        plt.close()
        
        plt.plot(loss_cr_arr)
        plt.xlabel("Epoch")
        plt.ylabel("CR Error")
        #plt.yscale("log")
        plt.title(f"{loss_cr_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/cr_loss.png")
        plt.clf()
        plt.close()
    
    def train(
        self,
        lb=-1, ub=0, imag_lb=-0.5, imag_ub=0.5,
        full_gradient=False, theta_eval=None,
        batch_size=500,

        # Joint phase
        joint_rounds=None,

        # Individual phase
        individual_rounds=None,
        # example:
        # individual_rounds = [
        #   dict(epochs=800,  lr=5e-6, T0=200, eta_min=1e-7),
        #   dict(epochs=1200, lr=2e-6, T0=300, eta_min=1e-8),
        # ]

        lam_monotone=0.0,
        lam_CR=1e-2,
        lam_growth=0.0,
        anchor_set=None,
    ):
        if full_gradient:
            assert theta_eval is not None
        
        if joint_rounds is None:
            joint_rounds = []

        if individual_rounds is None:
            individual_rounds = []

        # --------------------------------------------------
        # Logging containers
        # --------------------------------------------------
        total_loss_arr = []
        log_mse_arr = []
        bar_mse_arr = []
        cr_loss_arr = []

        # --------------------------------------------------
        # Joint training phase
        # --------------------------------------------------
        for r, cfg in enumerate(joint_rounds):
            epochs = cfg["epochs"]
            lr = cfg["lr"]
            T0 = cfg.get("T0", epochs)
            eta_min = cfg.get("eta_min", 0.0)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T0,
                T_mult=1,
                eta_min=eta_min,
            )

            for epoch in tqdm(range(epochs), desc="Joint training"):
                theta = theta_eval.clone() if full_gradient else \
                        self.sample_vector(lb, ub, imag_lb, imag_ub, batch_size)

                output = self.model(theta)
                log_phi_theta = output[:, 0:1]
                log_phi_i_theta = output[:, 1:]
                phi_theta = torch.exp(log_phi_theta)
                phi_i_theta = torch.exp(log_phi_i_theta)

                # BAR loss in log space
                bar_mse = self.bar_loss(theta, phi_theta, phi_i_theta)

                # Penalties
                cr = self.cauchy_riemann_penalty(self.model, theta) if lam_CR > 0 else 0.0
                loss = bar_mse + lam_CR * cr

                if lam_monotone > 0:
                    loss += lam_monotone * self.monotonicity_penalty(self.model, theta)
                if lam_growth > 0:
                    loss += lam_growth * self.growth_penalty(self.model, theta)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss_arr.append(loss.item())
                bar_mse_arr.append(bar_mse.item())
                cr_loss_arr.append(cr.item() if torch.is_tensor(cr) else 0.0)

        # --------------------------------------------------
        # Individual training rounds
        # --------------------------------------------------
        for r, cfg in enumerate(individual_rounds):
            epochs = cfg["epochs"]
            lr = cfg["lr"]
            T0 = cfg.get("T0", epochs)
            eta_min = cfg.get("eta_min", 0.0)

            for k in range(self.d + 1):
                self.model.freeze_all()
                if k == 0:
                    self.model.unfreeze_interior()
                else:
                    self.model.unfreeze_boundary_i(k - 1)

                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=T0,
                    T_mult=1,
                    eta_min=eta_min,
                )

                for _ in tqdm(range(epochs), desc=f"Round {r}, component {k}"):
                    theta = theta_eval.clone() if full_gradient else \
                            self.sample_vector(lb, ub, imag_lb, imag_ub, batch_size)

                    output = self.model(theta)
                    log_phi_theta = output[:, 0:1]
                    log_phi_i_theta = output[:, 1:]
                    phi_theta = torch.exp(log_phi_theta)
                    phi_i_theta = torch.exp(log_phi_i_theta)

                    log_mse = self.log_bar_loss(
                        theta, log_phi_theta, log_phi_i_theta, train_idx=k
                    )
                    bar_mse = self.bar_loss(theta, phi_theta, phi_i_theta)

                    cr = self.cauchy_riemann_penalty(self.model, theta) if lam_CR > 0 else 0.0
                    loss = log_mse + lam_CR * cr

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    total_loss_arr.append(loss.item())
                    bar_mse_arr.append(bar_mse.item())
                    cr_loss_arr.append(cr.item() if torch.is_tensor(cr) else 0.0)

        # --------------------------------------------------
        # Plot losses
        # --------------------------------------------------
        def save_plot(data, name, ylabel):
            plt.plot(data)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.yscale("log")
            plt.title(name)
            plt.savefig(f"Plots/{self.dir}/{name}.png")
            plt.clf()
            plt.close()

        save_plot(total_loss_arr, "total_loss", "Total Loss")
        save_plot(bar_mse_arr, "bar_mse_loss", "BAR-MSE Loss")
        save_plot(cr_loss_arr, "cr_loss", "CR Loss")

    
    def get_first_moment(self):
        s0 = torch.zeros(
            1, self.d,
            dtype=torch.cdouble,
            device=self.device,
            requires_grad=True
        )
        M = self.model(s0)[:, 0]   # interior MGF only
        grad = torch.autograd.grad(
            M.real.sum(),
            s0,
            create_graph=False
        )[0]
        return grad.real.squeeze().tolist()
    
    def eval(self, x):
        x = x.to(device = self.device)
        with torch.no_grad():
            output = self.model(x)
            output = torch.exp(output)
        return output.cpu()
    
    def save(self):
        state_dict = {"model_state_dict": self.model.cpu().state_dict()}
        torch.save(state_dict, f"Models/{self.dir}/mgf.pt")
        self.model.to(device = self.device)
    
    def load(self):
        state_dict = torch.load(f"Models/{self.dir}/mgf.pt", map_location=self.device)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model.to(device = self.device)
    
    def plot_compare_heatmap(self, real_lb, real_ub, imag_lb, imag_ub, phi_theta, phi_theta_true, title):
        n = int(len(phi_theta) ** 0.5)
        # Compute global min and max for color scale
        vmin = min(phi_theta.min(), phi_theta_true.min())
        vmax = max(phi_theta.max(), phi_theta_true.max())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # ---- Left plot: phi_theta ----
        im1 = axes[0].imshow(
            phi_theta.reshape((n, n)),
            extent=(real_lb, real_ub, imag_lb, imag_ub),
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_title("Model")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im1, ax=axes[0])
        # ---- Right plot: phi_theta_true ----
        im2 = axes[1].imshow(
            phi_theta_true.reshape((n, n)),
            extent=(real_lb, real_ub, imag_lb, imag_ub),
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_title("Ground Truth")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(f"Plots/{self.dir}/heatmap_{title.lower().replace(' ', '_')}.png")
        plt.clf()
        plt.close()

    def plot_compare(self, phi_theta, phi_theta_true, title):
        min_val = min(torch.min(phi_theta).item(), torch.min(phi_theta_true).item())
        max_val = max(torch.max(phi_theta).item(), torch.max(phi_theta_true).item())
        plt.scatter(phi_theta, phi_theta_true)
        plt.axline((min_val, min_val), (max_val, max_val), color = "red")
        plt.xlabel("Model")
        plt.ylabel("Ground Truth")
        plt.title(title)
        plt.savefig(f"Plots/{self.dir}/{title.lower().replace(' ', '_')}.png")
        plt.clf()
        plt.close()
