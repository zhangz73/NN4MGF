import os
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

class FFNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64, scale_by_zero = False):
        super(FFNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, output_dim),
        )
        self.scale_by_zero = scale_by_zero

    def forward(self, x):
        raw = self.network(x)
        if self.scale_by_zero:
            zero_point = torch.zeros(1, x.shape[1])
            raw0 = self.network(torch.zeros_like(zero_point))
            output = torch.exp(raw - raw0)   # (1, d+1)
        else:
            output = torch.exp(raw)
        return output
    
class MGFNet(nn.Module):
    def __init__(self, d, hidden_dim = 64):
        super(MGFNet, self).__init__()
        self.d = d
        self.interior_network = FFNet(self.d, 1, hidden_dim = hidden_dim, scale_by_zero = True)
        self.boundary_networks = nn.ModuleList()
        for i in range(self.d):
            self.boundary_networks.append(FFNet(self.d-1, 1, hidden_dim = hidden_dim))

    def forward(self, x):
        phi = self.interior_network(x)
        phi_i = torch.zeros((x.shape[0], self.d))
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
    def __init__(self, d, mu, sigma, R, hidden_dim = 128, dir = "."):
        self.d = d
        self.MU = mu
        self.SIGMA = sigma
        self.R = R
        self.model = MGFNet(self.d, hidden_dim = hidden_dim).double()
        self.dir = dir
        self.engine = SobolEngine(dimension=d)
    
    # ---- Define monotonicity penalty ----
    def monotonicity_penalty(self, model, s):
        s.requires_grad_(True)
        M_pred = model(s)
        grad = torch.autograd.grad(M_pred.sum(), s, create_graph=True)[0]
        return torch.relu(-grad).mean()  # penalize negative slopes

    def sample_vector(self, lb = -1, ub = 0, batch_size = 100):
#        vec = (ub - lb) * torch.rand(batch_size, self.d) + lb
        vec = (ub - lb) * self.engine.draw(batch_size) + lb
        return vec.double()
    
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
        scale_factor = torch.abs(lhs)
        diff = diff / (scale_factor + 1e-8)
        return torch.mean(diff.pow(2))
    
    def train(self, lb = -1, ub = 0, full_gradient = False, theta_eval = None, batch_size = 500, num_epochs = 21000, num_joint_epochs = 10000, num_individual_epochs = 1000, joint_init_lr = 1e-3, joint_scheduler_T0 = 100, joint_scheduler_Tmult = 1, joint_scheduler_eta_min = 0, individual_init_lr = 1e-6, individual_scheduler_T0 = 500, individual_scheduler_Tmult = 1, individual_scheduler_eta_min = 0, lam_monotone = 0.1, anchor_set = None):
        if full_gradient:
            assert theta_eval is not None
        ## Training
        optimizer = optim.Adam(self.model.parameters(), lr = joint_init_lr)
        #scheduler = ExponentialLR(optimizer, gamma=0.99)
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
                theta = self.sample_vector(lb = lb, ub = ub, batch_size = batch_size)
            if anchor_set is not None and len(anchor_set) > 0:
                anchors = anchor_set.clone()
                theta = torch.cat([theta, anchors], dim = 0)
            output = self.model(theta)
            phi_theta = output[:,0]
            phi_i_theta = output[:,1:]

            loss = self.bar_loss(theta, phi_theta, phi_i_theta)
#            loss += lam_monotone * self.monotonicity_penalty(self.model, theta)
            loss_arr.append(loss.item())
        #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        ## Evaluation
        plt.plot(loss_arr)
        plt.xlabel("Epoch")
        plt.ylabel("Bar Loss")
        #plt.yscale("log")
        plt.title(f"{loss_arr[-1]:.2e}")
        plt.savefig(f"Plots/{self.dir}/loss.png")
        plt.clf()
        plt.close()
    
    def eval(self, x):
        with torch.no_grad():
            output = self.model(x)
        return output
    
    def save(self):
        state_dict = {"model_state_dict": self.model.cpu().state_dict()}
        torch.save(state_dict, f"Models/{self.dir}/mgf.pt")
    
    def load(self):
        state_dict = torch.load(f"Models/{self.dir}/mgf.pt")
        self.model.load_state_dict(state_dict["model_state_dict"])
    
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
