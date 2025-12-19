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
        half = num_features // 2

        freqs_x = torch.logspace(0, 1, half, dtype=torch.float64)
        freqs_y = torch.logspace(0, 2, half, dtype=torch.float64)
        self.register_buffer("freqs_x", freqs_x)
        self.register_buffer("freqs_y", freqs_y)

    def forward(self, x):
        # x: (N, 2)
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]

        xb = (2.0 * math.pi) * x_coord * self.freqs_x
        yb = (2.0 * math.pi) * y_coord * self.freqs_y

        return torch.cat([torch.sin(xb), torch.cos(xb), torch.sin(yb), torch.cos(yb)], dim=-1)

class LogGMFNet(nn.Module):
    def __init__(self, ff_m = 32, hidden_dim = 128, scale_by_zero = False, x_min = -1, x_max = 0, y_min = -1, y_max = 1):
        super().__init__()
        self.ff = FourierFeatures(ff_m)
        self.net = nn.Sequential(
            nn.Linear(ff_m * 2, hidden_dim),
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
        x_norm = torch.cat([x_coord, y_coord], dim=1)

        raw = self.net(self.ff(x_norm))
        raw = torch.complex(raw[:,0:1], raw[:,1:2])
        if self.scale_by_zero:
            x_zero = torch.zeros_like(x_coord, dtype=torch.double)
            y_zero = torch.zeros_like(y_coord, dtype=torch.double)
            x_zero = 2.0 * (x_zero - self.X_MIN) / (self.X_MAX - self.X_MIN) - 1.0
            y_zero = 2.0 * (y_zero - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) - 1.0
            zero_point = torch.cat([x_zero, y_zero], dim=1) #torch.zeros(1, 2, dtype = torch.double, device = x.device)
            raw0 = self.net(self.ff(zero_point))
            raw0 = torch.complex(raw0[:,0:1], raw0[:,1:2])
#            output = torch.exp(raw - raw0)
            output = raw - raw0
        else:
#            output = torch.exp(raw)
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
        self.interior_network = LogGMFNet(ff_m = 32, hidden_dim = hidden_dim, scale_by_zero = True, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max) #PolyResNet(self.d, depth = 1, scale_by_zero = True)
        self.boundary_networks = nn.ModuleList()
        for i in range(self.d):
#            self.boundary_networks.append(PolyResNet(self.d-1, depth = 2, scale_by_zero = False))
            self.boundary_networks.append(FFNet(self.d-1, 1, hidden_dim = hidden_dim, omega_0 = omega_0))

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
        real_part = (ub - lb) * torch.rand(batch_size, 1, device = self.device) + lb
        real_part = real_part.double().to(device = self.device)

        # Draw imaginary part independently
        imag_part = (imag_ub - imag_lb) * torch.rand(batch_size, 1, device = self.device) + imag_lb
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
#        scale_factor = torch.abs(lhs)
#        diff = diff / (scale_factor + 1e-8)
        return torch.mean(torch.abs(diff) ** 2)
    
    def train_from_target(self, target_mgf_func, full_gradient = False, theta_eval = None, lb = -1, ub = 0, imag_lb=-0.5, imag_ub=0.5, batch_size = 500, num_epochs = 10000, init_lr = 1e-3, lam_monotone = 0.1, lam_CR = 1e-3, lam_growth = 1e-4):
        if full_gradient:
            assert theta_eval is not None
        optimizer = optim.Adam(self.model.parameters(), lr = init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
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
            else:
                theta = self.sample_vector(lb = lb, ub = ub, imag_lb=imag_lb, imag_ub=imag_ub, batch_size = batch_size)
            if theta_eval is not None:
                anchors = theta_eval.clone()
                # N = anchors.shape[0]
                # idx = torch.randint(0, N, size=(batch_size,), device=anchors.device)
                # anchors = anchors[idx]
                anchors = anchors.to(device = self.device)
                theta = torch.cat([theta, anchors], dim = 0)
            output = self.model(theta)
            log_phi_theta = output[:,0].view((-1, 1))
            phi_theta = torch.exp(log_phi_theta)
            phi_i_theta = output[:,1:].view((-1, self.d))
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
    
    def train(self, lb = -1, ub = 0, imag_lb=-0.5, imag_ub=0.5, full_gradient = False, theta_eval = None, batch_size = 500, num_epochs = 21000, num_joint_epochs = 10000, num_individual_epochs = 1000, joint_init_lr = 1e-3, joint_scheduler_T0 = 100, joint_scheduler_Tmult = 1, joint_scheduler_eta_min = 0, individual_init_lr = 1e-6, individual_scheduler_T0 = 500, individual_scheduler_Tmult = 1, individual_scheduler_eta_min = 0, lam_monotone = 0.1, lam_CR = 1e-3, lam_growth = 1e-4, anchor_set = None):
        if full_gradient:
            assert theta_eval is not None
        ## Training
        optimizer = optim.Adam(self.model.parameters(), lr = joint_init_lr)
#        scheduler = ExponentialLR(optimizer, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=joint_scheduler_T0,       # number of steps before first restart
            T_mult=joint_scheduler_Tmult,     # how much T increases after restart
            eta_min=joint_scheduler_eta_min #1e-6  # minimum LR
        )
        loss_arr = []
        prev_k = -1
        for epoch in tqdm(range(num_epochs)):
            if epoch >= num_joint_epochs:
                k = ((epoch - 1 - num_joint_epochs) // num_individual_epochs) % (self.d + 1)   # network index to train
                if k != prev_k:
                    # freeze all parameters
                    self.model.freeze_all()
                    # unfreeze only the chosen one
                    if k == 0:
                        self.model.unfreeze_interior()
                    else:
                        self.model.unfreeze_boundary_i(k-1)
                    prev_k = k
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = individual_init_lr)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=individual_scheduler_T0,       # number of steps before first restart
                        T_mult=individual_scheduler_Tmult,     # how much T increases after restart
                        eta_min=individual_scheduler_eta_min #1e-6  # minimum LR
                    )
            # ------------------------------------------------------
            # Normal training code
            # ------------------------------------------------------
            if full_gradient:
                theta = theta_eval.clone()
            else:
                theta = self.sample_vector(lb = lb, ub = ub, imag_lb=imag_lb, imag_ub=imag_ub, batch_size = batch_size)
            if anchor_set is not None and len(anchor_set) > 0:
                anchors = anchor_set.clone()
                anchors.to(device = self.device)
                theta = torch.cat([theta, anchors], dim = 0)
            output = self.model(theta)
            phi_theta = output[:,0].view((-1, 1))
            phi_i_theta = output[:,1:].view((-1, self.d))

            loss = self.bar_loss(theta, phi_theta, phi_i_theta)
            if lam_monotone > 0:
                loss += lam_monotone * self.monotonicity_penalty(self.model, theta)
            if lam_CR > 0:
                loss += lam_CR * self.cauchy_riemann_penalty(self.model, theta)
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
        plt.ylabel("Training Loss")
        #plt.yscale("log")
        plt.title(f"{loss_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/loss.png")
        plt.clf()
        plt.close()
    
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
            output[:,0] = torch.exp(output[:,0])
        return output.cpu()
    
    def save(self):
        state_dict = {"model_state_dict": self.model.cpu().state_dict()}
        torch.save(state_dict, f"Models/{self.dir}/mgf.pt")
        self.model.to(device = self.device)
    
    def load(self):
        state_dict = torch.load(f"Models/{self.dir}/mgf.pt", map_location=self.device)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model.to(device = self.device)
    
    def plot_compare_heatmap(self, lb, ub, phi_theta, phi_theta_true, title):
        n = int(len(phi_theta) ** 0.5)
        # Compute global min and max for color scale
        vmin = min(phi_theta.min(), phi_theta_true.min())
        vmax = max(phi_theta.max(), phi_theta_true.max())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # ---- Left plot: phi_theta ----
        im1 = axes[0].imshow(
            phi_theta.reshape((n, n)),
            extent=(lb, ub, lb, ub),
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
            extent=(lb, ub, lb, ub),
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
