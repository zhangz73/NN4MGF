import os
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

from fit_mgf import MGFTrainer
from inverse_laplace import InverseLaplace

d = 2
TRAIN_LB = -1
TRAIN_UB = 0
TRAIN_IMAG_LB = -0.5
TRAIN_IMAG_UB = 0.5
EVAL_LB = -30
EVAL_UB = 0
RETRAIN = True

MU1 = -1.

R = torch.tensor([
    [1., 0.],
    [-1., 1.]
])

SIGMA = torch.tensor([
    [1., 0.],
    [0., 1.]
])

MU = torch.tensor([MU1, 0.])

scheme = f"d=2/harrison"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)

def density(x1, x2):
    C = math.pi ** (-0.5) * (2 * abs(MU1)) ** 1.5
    r = (x1 ** 2 + x2 ** 2) ** 0.5
    theta = math.acos(x1 / r)
    rho = C * r ** (-0.5) * math.e ** (MU1 * (r+x1)) * math.cos(theta/2)
    return rho

def tail_prob(t):
    ## P(X1 + X2 > t) = 1 - P(X1 + X2 <= t)
    ##                = 1 - \int_0^t \int_0^{t-x1} \rho(x1, x2) dx2 dx1
    val = mp.quad(lambda x1: mp.quad(lambda x2: density(x1,x2), [0, t-x1]), [0, t])
    return float(1.0 - val)

def tail_prob_predicted(model, t):
    def tail_transform(s):
        if s == 0:
            return 1.0
        s_lst = torch.complex(torch.tensor([float(s.real)]), torch.tensor([float(s.imag)]))
        val = laplace_2d_to_xsum(model, s_lst).tolist()[0]
        return (1 - val) / s

    # Invert this new transform directly
    return mp.invertlaplace(tail_transform, t, method="dehoog") #"stehfest" #"cohen"

def laplace_xsum(s):
    ## L(s) = \int_0^{\inf} f(t) e^{-st} dt
    ##      = \int_0^{\inf} e^{-st} \int_0^t \int_0^{t-x1} \rho(x1, x2) dx1 dx2 dt
    ##      = 1/s \int_0^{inf} \int_0^{\inf} \rho(x1, x2) e^{-s(x1 + x2)} dx1 dx2
    # integrand for the 2D integral
    if float(s) == 0:
        return 1.0
    def integrand(x1, x2):
        return density(x1, x2) * mp.exp(-s * (x1 + x2))

    # do the double integral over [0, inf) x [0, inf)
    val = mp.quad(lambda x1: mp.quad(lambda x2: integrand(x1,x2), [0, mp.inf]), [0, mp.inf])
    return float(val / s)

## Assume s is a 1-d pytorch tensor
def laplace_2d_to_xsum(model, s_lst):
    ## Given L(s1, s2) = \int_0^{\inf} \int_0^{\inf} \rho(x1, x2) e^{-s1 x1 - s2 x2} dx1 dx2
    ## Want L(s) = \int_0^{\inf} f(t) e^{-st} dt, where t = x1 + x2
    ##           = \int_0^{\inf} e^{-st} \int_0^t \int_0^{t-x1} \rho(x1, x2) dx2 dx1 dt
    ##           = 1/s \int_0^{inf} \int_0^{\inf} \rho(x1, x2) e^{-s(x1 + x2)} dx1 dx2
    ##           = 1/s L(s, s)
    batch_size = len(s_lst)
    input = torch.zeros((batch_size, 2), dtype=torch.cdouble)
    input[:,0] = s_lst
    input[:,1] = s_lst
    with torch.no_grad():
        output = model.eval(-input)
        joint_laplace = output[:,0]
    ans = torch.empty_like(joint_laplace)
    # Case 1: s != 0
    mask = s_lst != 0
    ans[mask] = joint_laplace[mask] / s_lst[mask]
    # Case 2: s = 0 → Laplace transform must equal 1
    ans[~mask] = 1.0
    return ans

def create_lattice(lb, ub, n_points_per_dim = 50):
    x = np.linspace(lb, ub, n_points_per_dim)
    y = np.linspace(lb, ub, n_points_per_dim)
    X, Y = np.meshgrid(x, y)
    lattice = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis = 1)).double()
    return lattice

## Training
mgf_trainer = MGFTrainer(d = d, mu = MU, sigma = SIGMA, R = R, hidden_dim = 128, dir = f"{scheme}")
if RETRAIN:
    anchor_set = None #create_lattice(-200, -180, n_points_per_dim = 20)
    mgf_trainer.train(lb = TRAIN_LB - 0.5, ub = TRAIN_UB, imag_lb = TRAIN_IMAG_LB, imag_ub = TRAIN_IMAG_UB, full_gradient = False, theta_eval = None, batch_size = 1000, num_epochs = 21000, num_joint_epochs = 10000, num_individual_epochs = 1000, joint_init_lr = 1e-4, joint_scheduler_T0 = 100, joint_scheduler_Tmult = 1, joint_scheduler_eta_min = 0, individual_init_lr = 1e-7, individual_scheduler_T0 = 100, individual_scheduler_Tmult = 1, individual_scheduler_eta_min = 0, lam_monotone = 1e-1, lam_CR = 1e-4, lam_growth = 1e-4, anchor_set = anchor_set)
    mgf_trainer.save()
else:
    mgf_trainer.load()

## Comparing Tail probability of X1 + X2 against ground truth
t_lst = list(range(1, 6))
true_prob_lst = []
predicted_prob_lst = []
for t in tqdm(t_lst):
    ans = tail_prob(t)
    true_prob_lst.append(ans)
    predicted = tail_prob_predicted(mgf_trainer, t)
    predicted_prob_lst.append(predicted)

plt.scatter(t_lst, true_prob_lst, label = "Ground Truth", color = "red")
plt.plot(t_lst, predicted_prob_lst, label = "Predicted")
plt.legend()
plt.xlabel("t")
plt.ylabel("P(X1 + X2 > t)")
plt.title(f"Trained on $\\theta$ $\\in$ [{TRAIN_LB}, {TRAIN_UB}]^2")
plt.savefig(f"Plots/{scheme}/tail_prob.png")
plt.clf()
plt.close()

## Comparing Laplace transform of X1 + X2 against ground truth
s_lst = torch.linspace(-EVAL_UB, -EVAL_LB, steps = 10)[1:]
true_laplace_lst = []
for s in tqdm(s_lst):
    ans = laplace_xsum(float(s))
    true_laplace_lst.append(ans)
predicted_laplace_lst = laplace_2d_to_xsum(mgf_trainer, s_lst).real.tolist()

plt.scatter(s_lst, true_laplace_lst, label = "Ground Truth", color = "red")
plt.plot(s_lst, predicted_laplace_lst, label = "Predicted")
plt.legend()
plt.xlabel("s")
plt.ylabel("Laplace Transform of X1 + X2")
plt.title(f"Trained on $\\theta$ $\\in$ [{TRAIN_LB}, {TRAIN_UB}]^2")
plt.savefig(f"Plots/{scheme}/joint_laplace.png")
plt.clf()
plt.close()
