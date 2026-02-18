import os
import sys
import math
import mpmath as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Callable

from fit_mgf import MGFTrainer
from inverse_laplace import InverseLaplace

d = 10#25
TRAIN_LB = -5
TRAIN_UB = 0.5
TRAIN_IMAG_LB = -5
TRAIN_IMAG_UB = 5
EVAL_LB = -5
EVAL_UB = 0.5
EVAL_IMAG_LB = -5
EVAL_IMAG_UB = 5
RETRAIN = True

scheme = f"d={d}/skewed_symmetry"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)

MIN_REAL, MAX_REAL = float("inf"), -float("inf")
MIN_IMAG, MAX_IMAG = float("inf"), -float("inf")

THETA_LST = []

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
    alpha = alpha.to(device = theta.device)
    n = theta.shape[0]
    alpha_ratios = alpha / (alpha - theta)
    phi_theta = torch.prod(alpha_ratios, dim = 1)
    phi_i_theta = torch.zeros((n, d), dtype=torch.cdouble, device = theta.device)
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

def compute_true_moments(N):
    alpha = compute_alpha()
    moments = torch.zeros((N, d))
    curr = 1
    for n in range(N):
        curr *= (n+1) / alpha
        moments[n,:] = curr
    return moments.tolist()
    
## For 2d verification only
def density(x1, x2):
    alpha = compute_alpha()
    alpha = alpha.tolist()
    density = alpha[0] * math.e ** (-alpha[0] * x1) * alpha[1] * math.e ** (-alpha[1] * x2)
    return density

#def tail_prob(t):
#    ## P(X1 + X2 > t) = 1 - P(X1 + X2 <= t)
#    ##                = 1 - \int_0^t \int_0^{t-x1} \rho(x1, x2) dx2 dx1
#    val = mp.quad(lambda x1: mp.quad(lambda x2: density(x1,x2), [0, t-x1]), [0, t])
#    return float(1.0 - val)

def tail_prob(t):
    """
    Tail P(sum Exp(alpha_i) > t) for distinct alpha_i.
    alpha: list/tuple of distinct positive rates
    """
    alpha = compute_alpha().tolist()
    alpha = [mp.mpf(a) for a in alpha]
    t = mp.mpf(t)
    d = len(alpha)

    tail = mp.mpf("0")
    for i in range(d):
        ai = alpha[i]
        ci = mp.mpf("1")
        for j in range(d):
            if j == i:
                continue
            aj = alpha[j]
            ci *= aj / (aj - ai)
        tail += ci * mp.e**(-ai * t)
    return tail  # mp.mpf

#def improved_talbot_eq16(F: Callable[[mp.mpf], mp.mpf], t: mp.mpf, N: int) -> mp.mpf:
#    """
#    z(θ) = M/t(0.1446 + 3.0232θ^2 / (θ^2 - 3.0767π^2) + 0.2339iθ)
#    z'(θ) = M/t (-(2*3.0232*(3.0767π^2)θ)/(θ^2-3.0767π^2)^2 + 0.2339i)
#    f ≈ 1/(Mi) Σ_{θ=-π+(2k-1)π/M} exp(z(θ)t) F(z(θ)) z'(θ)
#    """
#    t = mp.mpf(t)
#    if N <= 1:
#        raise ValueError("N must be >= 2")
#
#    total = mp.mpf("0.0")
#    for k in range(1, N+1):
#        theta = -mp.pi + (2*k-1) * mp.pi/N
#        z = N/t * (0.1446 + ((3.0232*theta**2)/(theta**2 - 3.0767*mp.pi**2)) + 0.2339j * theta)
#        z_prime = N/t * (-(2*3.0232*(3.0767*mp.pi**2)*theta)/(theta**2 - 3.0767*mp.pi**2)**2 + 0.2339j)
#        term = mp.e ** (z * t) * F(z) * z_prime
#        total += term
#    return (1 / (N * 1j)) * total

def improved_talbot_eq16(F, t, N):
    """
    z(θ) = M/t(0.1446 + 3.0232θ^2 / (θ^2 - 3.0767π^2) + 0.2339iθ)
    z'(θ) = M/t (-(2*3.0232*(3.0767π^2)θ)/(θ^2-3.0767π^2)^2 + 0.2339i)
    f ≈ 1/(Mi) Σ_{θ=-π+(2k-1)π/M} exp(z(θ)t) F(z(θ)) z'(θ)
    """
    if N <= 1:
        raise ValueError("N must be >= 2")

    total = 0.0
    for k in range(1, N+1):
        theta = -math.pi + (2*k-1) * math.pi/N
        z = N/t * (0.1446 + ((3.0232*theta**2)/(theta**2 - 3.0767*math.pi**2)) + 0.2339j * theta)
        z_prime = N/t * (-(2*3.0232*(3.0767*math.pi**2)*theta)/(theta**2 - 3.0767*math.pi**2)**2 + 0.2339j)
        term = math.e ** (z * t) * F(z) * z_prime
        total += term
    return (1 / (N * 1j)) * total

def talbot_1(F, t, N):
    if N <= 1:
        raise ValueError("N must be >= 2")
    
    alpha = 1
    s = 0.7556
    m = 0.8597
    n = 0.3029

    total = 0.0
    for k in range(-N, N):
        theta = (2*k+1) * math.pi/(2*N)
        z = N/t * (-s + m * (1 + (2 * theta ** 2) / (theta ** 2 - alpha * math.pi ** 2) + n * 1j * theta))
        z_prime = N/t * m * (-(4 * alpha * math.pi ** 2 * theta) / (theta ** 2 - alpha * math.pi ** 2) ** 2 + n * 1j)
        term = math.e ** (z * t) * F(z) * z_prime
        total += term
    return (1 / (N * 2j)) * total

def talbot_2(F, t, N):
    if N <= 1:
        raise ValueError("N must be >= 2")
    
    alpha = 0.7
    s = 0.4814
    m = 0.6443
    n = 0.5653
    
    def cot(x):
        return math.cos(x) / math.sin(x)
    
    def csc(x):
        return 1 / math.sin(x)

    total = 0.0 + 0.0j
    for k in range(-N, N):
        theta = (2*k+1) * np.pi/(2*N)
        z = N/t * (-s + m * (theta * cot(alpha * theta) + n * 1j * theta))
        z_prime = N/t * m * (cot(alpha * theta) - alpha * theta * csc(alpha * theta) ** 2 + n * 1j)
        term = math.e ** (z * t) * F(z) * z_prime
        total += term
    return (1 / (N * 2j)) * total

def tail_prob_predicted(model, t):
    def tail_transform(s):
        global MIN_REAL, MAX_REAL, MIN_IMAG, MAX_IMAG
        MIN_REAL = min(MIN_REAL, float(s.real))
        MAX_REAL = max(MAX_REAL, float(s.real))
        MIN_IMAG = min(MIN_IMAG, float(s.imag))
        MAX_IMAG = max(MAX_IMAG, float(s.imag))
        if s == 0:
            return 1.0
        s_lst = torch.complex(torch.tensor([float(s.real)]), torch.tensor([float(s.imag)]))
        val = laplace_2d_to_xsum(model, s_lst).tolist()[0]
        return (1 - val) / s

    # Invert this new transform directly
    # improved_talbot_eq16(tail_transform, t, N = 5)
    return mp.invertlaplace(tail_transform, t, method="talbot", degree = 5) #"stehfest" #"cohen" #talbot_2(tail_transform, t, N = 5) #

#def laplace_xsum(s):
#    ## L(s) = \int_0^{\inf} f(t) e^{-st} dt
#    ##      = \int_0^{\inf} e^{-st} \int_0^t \int_0^{t-x1} \rho(x1, x2) dx1 dx2 dt
#    ##      = 1/s \int_0^{inf} \int_0^{\inf} \rho(x1, x2) e^{-s(x1 + x2)} dx1 dx2
#    # integrand for the 2D integral
#    global MIN_REAL, MAX_REAL, MIN_IMAG, MAX_IMAG
#    MIN_REAL = min(MIN_REAL, float(s.real))
#    MAX_REAL = max(MAX_REAL, float(s.real))
#    MIN_IMAG = min(MIN_IMAG, float(s.imag))
#    MAX_IMAG = max(MAX_IMAG, float(s.imag))
#    if s == 0:
#        return 1.0
#    def integrand(x1, x2):
#        return density(x1, x2) * mp.exp(-s * (x1 + x2))
#
#    # do the double integral over [0, inf) x [0, inf)
#    val = mp.quad(lambda x1: mp.quad(lambda x2: integrand(x1,x2), [0, mp.inf]), [0, mp.inf])
#    return val
def laplace_xsum(s):
    if s == 0:
        return 1.0
    alpha = compute_alpha().tolist()

    s = mp.mpc(s)
    val = mp.mpf("1.0")
    for a in alpha:
        a = mp.mpf(a)
        val *= a / (a + s)
    return val

## Assume s is a 1-d pytorch tensor
def laplace_2d_to_xsum(model, s_lst):
    ## Given L(s1, s2) = \int_0^{\inf} \int_0^{\inf} \rho(x1, x2) e^{-s1 x1 - s2 x2} dx1 dx2
    ## Want L(s) = \int_0^{\inf} f(t) e^{-st} dt, where t = x1 + x2
    ##           = \int_0^{\inf} e^{-st} \int_0^t \int_0^{t-x1} \rho(x1, x2) dx2 dx1 dt
    ##           = 1/s \int_0^{inf} \int_0^{\inf} \rho(x1, x2) e^{-s(x1 + x2)} dx1 dx2
    ##           = 1/s L(s, s)
    global THETA_LST
    batch_size = len(s_lst)
    input = torch.zeros((batch_size, d), dtype=torch.cdouble)
    input[:,:] = s_lst.reshape((-1, 1))
    THETA_LST.append(-input)
    with torch.no_grad():
        output = model.eval(-input)
        joint_laplace = output[:,0]
#    ans = torch.empty_like(joint_laplace)
#    # Case 1: s != 0
#    mask = s_lst != 0
#    ans[mask] = joint_laplace[mask] / s_lst[mask]
#    # Case 2: s = 0 → Laplace transform must equal 1
#    ans[~mask] = 1.0
    ans = joint_laplace
    return ans

def print_table(file, t_lst, true_lst, predicted_lst):
    assert len(t_lst) == len(true_lst) == len(predicted_lst)
    true_lst = [float(x) for x in true_lst]
    predicted_lst = [float(x) for x in predicted_lst]

    header = (
        f"{'t':>12} | "
        f"{'true value':>14} | "
        f"{'pred value':>14} | "
        f"{'abs error':>14} | "
        f"{'rel error':>14}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)
    file.write(header + "\n")
    file.write(sep + "\n")

    for t, true_val, pred_val in zip(t_lst, true_lst, predicted_lst):
        abs_err = abs(pred_val - true_val)
        rel_err = abs_err / true_val if true_val != 0 else float("nan")
        row = (
            f"{t:12.4e} | "
            f"{true_val:14.6e} | "
            f"{pred_val:14.6e} | "
            f"{abs_err:14.6e} | "
            f"{rel_err:14.6e}"
        )

        print(row)
        file.write(row + "\n")


## Generate problem instance
R, SIGMA, MU = generate_instance(d = d)
alpha = compute_alpha()
print("alpha:", alpha)
print("R:", R)
print("Sigma:", SIGMA)
print("Mu:", MU)

excluded_boxes = []
#for real_pole in set(alpha):
#    excluded_boxes.append((float(real_pole) - 1, float(real_pole) + 1, -1, 1))

mgf_trainer = MGFTrainer(d = d, mu = MU, sigma = SIGMA, R = R, hidden_dim = 128, dir = f"{scheme}", x_min = TRAIN_LB, x_max = TRAIN_UB, y_min = TRAIN_IMAG_LB, y_max = TRAIN_IMAG_UB)

## Generate evaluation data
if False: #d == 2:
    n_points_per_dim = 50
    x = np.linspace(LB, UB, n_points_per_dim)
    y = np.linspace(LB, UB, n_points_per_dim)
    X, Y = np.meshgrid(x, y)
    theta_eval = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis = 1)).float()
else:
    theta_eval = mgf_trainer.sample_vector(lb=EVAL_LB, ub=EVAL_UB, imag_lb=EVAL_IMAG_LB, imag_ub=EVAL_IMAG_UB, excluded_boxes = excluded_boxes, batch_size = 10000)

if RETRAIN:
    anchor_set = None
    joint_rounds = [
        # Warm-up / coarse fit
        dict(epochs=400000, lr=1e-3, T0=400000, eta_min=1e-4),
        # Refine BAR fit
        dict(epochs=400000, lr=1e-4, T0=400000, eta_min=1e-5),
#        # Final polishing
        #dict(epochs=20000, lr=3e-5, T0=20000, eta_min=1e-5),
        #dict(epochs=20000, lr=1e-5, T0=20000, eta_min=3e-6),
    ]
    individual_rounds = [
        dict(epochs=800, lr=2e-4, T0=800, eta_min=2e-5),
        dict(epochs=800, lr=7e-5, T0=800, eta_min=7e-6),
    ]
    lam_monotone = 1e1
    lam_CR = 1e1
    lam_growth = 0
    lam_zero_anchor = 1e-1
    lam_boundary_consistent = 1e1
    batch_size = 16384 #4096 #8192#16384
    max_grad_sample = 1024
    max_boundary_grad_sample = 128
    train_freq = 1
#    joint_rounds = [
#        dict(epochs=5000, lr=1e-3, T0=5000, eta_min=1e-5),
##        dict(epochs=5000, lr=1e-4, T0=5000, eta_min=1e-6),
##        dict(epochs=500, lr=1e-5, T0=5000, eta_min=1e-8),
#    ]
#    individual_rounds = [
#        dict(epochs=2000,  lr=1e-3, T0=5000, eta_min=1e-6)
#    ] * 1
    individual_rounds = None
    mgf_trainer.train(lb = TRAIN_LB, ub = TRAIN_UB, imag_lb = TRAIN_IMAG_LB, imag_ub = TRAIN_IMAG_UB, excluded_boxes = excluded_boxes, full_gradient = False, theta_eval = None, batch_size = batch_size, max_grad_sample = max_grad_sample, max_boundary_grad_sample = max_boundary_grad_sample, train_freq = train_freq, joint_rounds = joint_rounds, individual_rounds = individual_rounds, lam_monotone = lam_monotone, lam_CR = lam_CR, lam_growth = lam_growth, lam_zero_anchor = lam_zero_anchor, lam_boundary_consistent = lam_boundary_consistent, anchor_set = anchor_set)
    mgf_trainer.save()
else:
    mgf_trainer.load()

## Comparing against ground truth
phi_theta_true, phi_i_theta_true = compute_true_phi(theta_eval)
with torch.no_grad():
    output = mgf_trainer.eval(theta_eval)
    output = output.to(device = theta_eval.device)
    phi_theta = output[:,0]
    phi_i_theta = output[:,1:]
print("Bar Loss (Model):", mgf_trainer.bar_loss(theta_eval, phi_theta, phi_i_theta))
print("Bar Loss (Truth):", mgf_trainer.bar_loss(theta_eval, phi_theta_true, phi_i_theta_true))
phi_theta, phi_i_theta = phi_theta.cpu(), phi_i_theta.cpu()
phi_theta_true, phi_i_theta_true = phi_theta_true.cpu(), phi_i_theta_true.cpu()
theta_eval = theta_eval.cpu()
mgf_trainer.plot_compare(phi_theta, phi_theta_true, title = "Interior")
if d == 2:
    mgf_trainer.plot_compare_heatmap(real_lb = EVAL_LB, real_ub = EVAL_UB, imag_lb = EVAL_IMAG_LB, imag_ub = EVAL_IMAG_UB, phi_theta = phi_theta, phi_theta_true = phi_theta_true, title = "Interior")
for i in range(d):
    mgf_trainer.plot_compare(phi_i_theta[:,i], phi_i_theta_true[:,i], title = f"Boundary {i}")
    if d == 2:
        mgf_trainer.plot_compare_heatmap(real_lb = EVAL_LB, real_ub = EVAL_UB, imag_lb = EVAL_IMAG_LB, imag_ub = EVAL_IMAG_UB, phi_theta = phi_i_theta[:,i], phi_theta_true = phi_i_theta_true[:,i], title = f"Boundary {i}")

## Compute first moment
#first_moment = mgf_trainer.get_first_moment()
#print("Mean queue lengths:", first_moment)
#print("True mean queue lengths:", [1/x for x in alpha])
moment_N = 5
predicted_moments = mgf_trainer.moments_from_mgf(N = moment_N)
true_moments = compute_true_moments(N=moment_N)

## Comparing Tail probability of X1 + X2 against ground truth
t_lst = list(range(1, 6))
true_prob_lst = []
predicted_prob_lst = []

with open(f"Plots/{scheme}/tables.txt", "w") as file:
    for n in range(moment_N):
        file.write(f"Moment #{n+1}:\n")
        d_lst = list(range(d))
        true_moment_lst = true_moments[n]
        pred_moment_lst = predicted_moments[n]
        print_table(file, d_lst, true_moment_lst, pred_moment_lst)
        file.write("\n\n")
    
for t in tqdm(t_lst):
    ans = tail_prob(t)
    true_prob_lst.append(float(ans))
    predicted = tail_prob_predicted(mgf_trainer, t)
    predicted_prob_lst.append(predicted)

with open(f"Plots/{scheme}/tables.txt", "a") as file:
    print("Tail probabilities:")
    file.write("Tail probabilities:\n")
    print_table(file, t_lst, true_prob_lst, predicted_prob_lst)
    file.write("\n\n")

plt.scatter(t_lst, true_prob_lst, label = "Ground Truth", color = "red")
plt.plot(t_lst, predicted_prob_lst, label = "Predicted")
plt.legend()
plt.xlabel("t")
plt.ylabel(r"P($\sum_i$ $X_i$ > t)")
plt.title(f"Trained on $\\theta$ $\\in$ [{TRAIN_LB}, {TRAIN_UB}]^2")
plt.savefig(f"Plots/{scheme}/tail_prob.png")
plt.clf()
plt.close()

print("Real range:", MIN_REAL, MAX_REAL)
print("Imag range:", MIN_IMAG, MAX_IMAG)

## Comparing Laplace transform of X1 + X2 against ground truth
s_lst = torch.linspace(0, 5, steps = 6)[1:]
true_laplace_lst = []
for s in tqdm(s_lst):
    ans = laplace_xsum(float(s))
    true_laplace_lst.append(float(ans.real))
predicted_laplace_lst = laplace_2d_to_xsum(mgf_trainer, s_lst).real.tolist()

with open(f"Plots/{scheme}/tables.txt", "a") as file:
    print("Laplace values:")
    file.write("Laplace values:\n")
    print_table(file, s_lst, true_laplace_lst, predicted_laplace_lst)

plt.scatter(s_lst, true_laplace_lst, label = "Ground Truth", color = "red")
plt.plot(s_lst, predicted_laplace_lst, label = "Predicted")
plt.legend()
plt.xlabel("s")
plt.ylabel("Laplace Transform of $\sum_i$ $X_i$")
plt.title(f"Trained on $\\theta$ $\\in$ [{TRAIN_LB}, {TRAIN_UB}]^2")
plt.savefig(f"Plots/{scheme}/joint_laplace.png")
plt.clf()
plt.close()
