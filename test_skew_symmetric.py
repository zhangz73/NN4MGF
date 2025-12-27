import os
import mpmath
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from fit_mgf import MGFTrainer
from inverse_laplace import InverseLaplace

d = 2
TRAIN_LB = -1
TRAIN_UB = 0
TRAIN_IMAG_LB = -1
TRAIN_IMAG_UB = 1
EVAL_LB = -1
EVAL_UB = 0
EVAL_IMAG_LB = -1
EVAL_IMAG_UB = 1
RETRAIN = True

scheme = f"d={d}/skewed_symmetry"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)

def check_skew_symmetry(R, SIGMA, MU):
    LHS = 2 * SIGMA
    RHS = R @ torch.inverse(torch.diag(torch.diagonal(R))) @ torch.diag(torch.diagonal(SIGMA)) + torch.diag(torch.diagonal(SIGMA)) @ torch.inverse(torch.diag(torch.diagonal(R))) @ R.T
    is_not_equiv = LHS != RHS
    assert torch.sum(is_not_equiv) == 0

def generate_instance(d = 2):
    c_vec = torch.ones(d + 1)
    beta_vec = torch.arange(d + 1) + 1.
    R = torch.zeros((d, d))
    SIGMA = torch.zeros((d, d))
    MU = torch.zeros(d)
    for i in range(d):
        R[i, i] = 1
        SIGMA[i, i] = c_vec[i] + c_vec[i+1]
        MU[i] = beta_vec[i] - beta_vec[i+1]
        if i > 0:
            R[i, i-1] = -1
            SIGMA[i, i-1] = -c_vec[i]
            SIGMA[i-1, i] = -c_vec[i]
    ## Check if the Skew Symmetry condition is satisfied
    check_skew_symmetry(R, SIGMA, MU)
    return R, SIGMA, MU

def compute_alpha():
    alpha_colvec = -2 * torch.inverse(torch.diag(torch.diagonal(SIGMA))) @ torch.diag(torch.diagonal(R)) @ torch.inverse(R) @ MU
    return alpha_colvec.flatten()

## Assume theta is N x d
def compute_true_phi(theta):
    alpha = compute_alpha()
    print("alpha:", alpha)
    n = theta.shape[0]
    alpha_ratios = alpha / (alpha - theta)
    phi_theta = torch.prod(alpha_ratios, dim = 1)
    phi_i_theta = torch.zeros((n, d), dtype=torch.cdouble)
    for i in range(d):
#        val = SIGMA[i,i] / (2 * R[i,i]) * alpha[i] * torch.prod(alpha_ratios, dim = 1) / (alpha_ratios[:,i])
        val = SIGMA[i,i] / (2 * R[i,i]) * phi_theta * (alpha[i] - theta[:,i])
        phi_i_theta[:,i] = val
    return phi_theta, phi_i_theta

## Assume y is N x d
def compute_true_density(y):
    alpha = compute_alpha()
    n = y.shape[0]
    density = torch.prod(alpha * torch.exp(-alpha * y), dim = 1)
    density_i = torch.zeros((n, d))
    for i in range(d):
        val = SIGMA[i,i] / (2 * R[i,i]) * density / torch.exp(-alpha[i] * y[:,i])
        density_i[:,i] = val
    return density, density_i

def compare_density(y_lst, density_predicted, density_true, title):
    plt.plot(y_lst, density_predicted, label = "Predicted")
    plt.plot(y_lst, density_true, label = "Ground Truth")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.savefig(f"Plots/{scheme}/{title.lower().replace(' ', '_')}.png")
    plt.clf()
    plt.close()

## Generate problem instance
R, SIGMA, MU = generate_instance(d = d)
print("R:", R)
print("Sigma:", SIGMA)
print("Mu:", MU)

mgf_trainer = MGFTrainer(d = d, mu = MU, sigma = SIGMA, R = R, hidden_dim = 128, dir = f"{scheme}")

## Generate evaluation data
if False: #d == 2:
    n_points_per_dim = 50
    x = np.linspace(LB, UB, n_points_per_dim)
    y = np.linspace(LB, UB, n_points_per_dim)
    X, Y = np.meshgrid(x, y)
    theta_eval = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis = 1)).float()
else:
    theta_eval = mgf_trainer.sample_vector(lb=EVAL_LB, ub=EVAL_UB, imag_lb=EVAL_IMAG_LB, imag_ub=EVAL_IMAG_UB, batch_size = 10000)

if RETRAIN:
    anchor_set = None
    joint_rounds = [
        dict(epochs=2000, lr=1e-3, T0=5000, eta_min=1e-6),
#        dict(epochs=500, lr=1e-4, T0=5000, eta_min=1e-7),
#        dict(epochs=500, lr=1e-5, T0=5000, eta_min=1e-8),
    ]
#    individual_rounds = [
#        dict(epochs=5000,  lr=1e-3, T0=5000, eta_min=1e-6)
#    ] * 3 + [
#        dict(epochs=5000, lr=1e-4, T0=5000, eta_min=1e-8)
#    ] * 3
    individual_rounds = None
    mgf_trainer.train(lb = TRAIN_LB, ub = TRAIN_UB, imag_lb = TRAIN_IMAG_LB, imag_ub = TRAIN_IMAG_UB, full_gradient = False, theta_eval = None, batch_size = 1024, joint_rounds = joint_rounds, individual_rounds = individual_rounds, lam_monotone = 1e-1, lam_CR = 1e-1, lam_growth = 0, anchor_set = anchor_set)
    mgf_trainer.save()
else:
    mgf_trainer.load()

## Comparing against ground truth
phi_theta_true, phi_i_theta_true = compute_true_phi(theta_eval)
with torch.no_grad():
    output = mgf_trainer.eval(theta_eval)
    phi_theta = output[:,0]
    phi_i_theta = output[:,1:]
print("Bar Loss (Model):", mgf_trainer.bar_loss(theta_eval, phi_theta, phi_i_theta))
print("Bar Loss (Truth):", mgf_trainer.bar_loss(theta_eval, phi_theta_true, phi_i_theta_true))
mgf_trainer.plot_compare(phi_theta, phi_theta_true, title = "Interior")
if d == 2:
    mgf_trainer.plot_compare_heatmap(real_lb = EVAL_LB, real_ub = EVAL_UB, imag_lb = EVAL_IMAG_LB, imag_ub = EVAL_IMAG_UB, phi_theta = phi_theta, phi_theta_true = phi_theta_true, title = "Interior")
for i in range(d):
    mgf_trainer.plot_compare(phi_i_theta[:,i], phi_i_theta_true[:,i], title = f"Boundary {i}")
    if d == 2:
        mgf_trainer.plot_compare_heatmap(real_lb = EVAL_LB, real_ub = EVAL_UB, imag_lb = EVAL_IMAG_LB, imag_ub = EVAL_IMAG_UB, phi_theta = phi_i_theta[:,i], phi_theta_true = phi_i_theta_true[:,i], title = f"Boundary {i}")

## Compare the density
#inverse_laplace = InverseLaplace(mgf_trainer, dps = 100, pretty = True)
#if d == 2:
#    y1_lst = np.arange(10) + 1
#    y2_lst = np.arange(10) + 1
#    y = torch.concat([torch.from_numpy(y1_lst).reshape((-1, 1)), torch.from_numpy(y2_lst).reshape((-1, 1))], dim = 1)
#    density, density_i = compute_true_density(y)
#    predicted_density_i = torch.zeros((len(y1_lst), 2))
#    for i in tqdm(range(len(y1_lst))):
#        predicted_density_i[i, 0] = inverse_laplace.invert(y1_lst[i], k = 1)
#        predicted_density_i[i, 1] = inverse_laplace.invert(y2_lst[i], k = 2)
#    print(predicted_density_i)
#    compare_density(y1_lst, predicted_density_i[:,0].numpy(), density_i[:,0].numpy(), title = f"Density Boundary {0}")
#    compare_density(y2_lst, predicted_density_i[:,1].numpy(), density_i[:,1].numpy(), title = f"Density Boundary {1}")
