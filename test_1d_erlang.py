import os
import math
import mpmath as mp
import numpy as np
import pandas as pd
from scipy.stats import gamma
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from fit_mgf import *
from inverse_laplace import InverseLaplace

d = 1
TRAIN_LB = -7
TRAIN_UB = 7
TRAIN_IMAG_LB = -5
TRAIN_IMAG_UB = 5
EVAL_LB = -7
EVAL_UB = 7
EVAL_IMAG_LB = -5
EVAL_IMAG_UB = 5
RETRAIN = True
S_LST = []

K = 2
LAM = 2

MIN_REAL, MAX_REAL = float("inf"), -float("inf")
MIN_IMAG, MAX_IMAG = float("inf"), -float("inf")

scheme = f"d={d}/erlang"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)

mp.dps = 100

def target_mgf_func(theta):
    mgf = (LAM / (LAM - theta)) ** K
    return mgf

def prob(t):
    pdf = gamma.pdf(t, a=K, scale=1.0/LAM)
    return pdf

def prob_predicted(model, t):
    def transform(s):
        global MIN_REAL, MAX_REAL, MIN_IMAG, MAX_IMAG
        MIN_REAL = min(MIN_REAL, float(s.real))
        MAX_REAL = max(MAX_REAL, float(s.real))
        MIN_IMAG = min(MIN_IMAG, float(s.imag))
        MAX_IMAG = max(MAX_IMAG, float(s.imag))
        if abs(s) == 0:
            return 1.0
        s = complex(s)
        s_lst = torch.tensor([[s]], dtype=torch.cdouble)
        with torch.no_grad():
            val = model.eval(-s_lst)[0, 0].item()
        return val

    # Invert this new transform directly
    return mp.invertlaplace(transform, t, method="dehoog")

def tail_prob(t):
    cdf = gamma.cdf(t, a=K, scale=1.0/LAM)
    return 1 - cdf

def tail_prob_predicted(model, t):
    def tail_transform(s):
        global MIN_REAL, MAX_REAL, MIN_IMAG, MAX_IMAG, S_LST
        MIN_REAL = min(MIN_REAL, float(s.real))
        MAX_REAL = max(MAX_REAL, float(s.real))
        MIN_IMAG = min(MIN_IMAG, float(s.imag))
        MAX_IMAG = max(MAX_IMAG, float(s.imag))
        S_LST.append(complex(s))
        if abs(s) == 0:
            return 1.0
        s = complex(s)
        s_lst = torch.tensor([[s]], dtype=torch.cdouble)
        with torch.no_grad():
            val = model.eval(-s_lst)[0, 0].item()
        return (1.0 - val) / s

    # Invert this new transform directly
    return mp.invertlaplace(tail_transform, t, method="talbot", degree=5)

## Generate evaluation data
n_points_per_dim = 20
theta_real_lst = np.linspace(EVAL_LB, EVAL_UB, n_points_per_dim)
theta_imag_lst = np.linspace(EVAL_IMAG_LB, EVAL_IMAG_UB, n_points_per_dim)
X, Y = np.meshgrid(theta_real_lst, theta_imag_lst)
theta_real_lst, theta_imag_lst = torch.from_numpy(X.ravel()).double(), torch.from_numpy(Y.ravel()).double()
theta_lst = torch.complex(theta_real_lst, theta_imag_lst)

## Training
mgf_trainer = MGFTrainer(d = 1, mu = None, sigma = None, R = None, hidden_dim = 128, dir = f"{scheme}", x_min = TRAIN_LB, x_max = TRAIN_UB, y_min = TRAIN_IMAG_LB, y_max = TRAIN_IMAG_UB)
if RETRAIN:
    t_lst = list(range(1, 6))
    for t in tqdm(t_lst):
        tail_prob_predicted(mgf_trainer, t)
    theta_eval = -torch.tensor(S_LST, dtype=torch.cdouble).reshape((-1, 1))
    mgf_trainer.train_from_target(target_mgf_func, full_gradient = False, theta_eval = theta_eval, lb = TRAIN_LB, ub = TRAIN_UB, imag_lb=TRAIN_IMAG_LB, imag_ub=TRAIN_IMAG_UB, batch_size = 4096, num_epochs = 20000, init_lr = 1e-3, lam_monotone = 0, lam_CR = 1e-1, lam_growth = 0)
    mgf_trainer.save()
    predicted_mgf_lst = mgf_trainer.eval(theta_eval.reshape((-1, 1)))[:,0]
    true_mgf_lst = target_mgf_func(theta_eval.flatten())
    diff_lst = torch.abs(predicted_mgf_lst - true_mgf_lst)
    for i in range(len(diff_lst)):
        theta = theta_eval[i,0]
        ans = true_mgf_lst[i]
        pred = predicted_mgf_lst[i]
        diff = diff_lst[i]
        print(f"theta = {theta}: True MGF = {ans}, Predicted MGF = {pred}, diff = {diff:.2e}")
else:
    mgf_trainer.load()

## Compute first moment
first_moment = mgf_trainer.get_first_moment()
print("True mean:", K/LAM)
print("Predicted mean from NN:", first_moment)

## Visualize ground truth MGF
true_mgf_lst = target_mgf_func(theta_lst)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(
    true_mgf_lst.real.reshape((n_points_per_dim, n_points_per_dim)),
    extent=[EVAL_LB, EVAL_UB, EVAL_IMAG_LB, EVAL_IMAG_UB],  # [xmin, xmax, ymin, ymax]
    origin='lower',  # ensures Y-axis goes from bottom to top
    aspect='auto',
    cmap='viridis',
)
axes[0].set_title("True MGF (Real Part)")
axes[0].set_xlabel("Re(s)")
axes[0].set_ylabel("Im(s)")
fig.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(
    true_mgf_lst.imag.reshape((n_points_per_dim, n_points_per_dim)),
    extent=[EVAL_LB, EVAL_UB, EVAL_IMAG_LB, EVAL_IMAG_UB],  # [xmin, xmax, ymin, ymax]
    origin='lower',  # ensures Y-axis goes from bottom to top
    aspect='auto',
    cmap='viridis',
)
axes[1].set_title("True MGF (Imag Part)")
axes[1].set_xlabel("Re(s)")
axes[1].set_ylabel("Im(s)")
fig.colorbar(im2, ax=axes[1])
plt.savefig(f"Plots/{scheme}/true_mgf.png")
plt.clf()
plt.close()

## Comparing MGF against ground truth
predicted_mgf_lst = mgf_trainer.eval(theta_lst.reshape((-1, 1)))[:,0]
true_mgf_lst = target_mgf_func(theta_lst)
diff_lst = torch.abs(predicted_mgf_lst - true_mgf_lst) / torch.abs(true_mgf_lst)
plt.imshow(
    diff_lst.reshape((n_points_per_dim, n_points_per_dim)),
    extent=[EVAL_LB, EVAL_UB, EVAL_IMAG_LB, EVAL_IMAG_UB],  # [xmin, xmax, ymin, ymax]
    origin='lower',  # ensures Y-axis goes from bottom to top
    aspect='auto',
    cmap='viridis',
    norm=LogNorm(vmin=diff_lst.min().item() + 1e-12, vmax=min(diff_lst.max().item(), 1e50))  # log scale
)
print(torch.mean(diff_lst))
plt.colorbar(label='Relative Error')
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.title('MGF Prediction Relative Error')
plt.savefig(f"Plots/{scheme}/mgf_error.png")
plt.clf()
plt.close()

## Compare tail probability of X against ground truth
t_lst = list(range(1, 6))
true_prob_lst = []
predicted_prob_lst = []
diff_lst = []
for t in tqdm(t_lst):
    ans = tail_prob(t)
    true_prob_lst.append(ans)
    predicted = tail_prob_predicted(mgf_trainer, t)
    predicted_prob_lst.append(predicted)
    diff = predicted - ans
    diff_lst.append(abs(diff))

print("| t | Truth      | Prediction          | Absolute Error |")
print("|---|------------|---------------------|----------------|")

for i in range(len(t_lst)):
    t = t_lst[i]
    ans = float(true_prob_lst[i])
    predicted = float(predicted_prob_lst[i])
    diff = float(diff_lst[i])
    print(f"| {t} | {ans:<10} | {predicted:.18f} | {diff:.2e} |")

plt.scatter(t_lst, true_prob_lst, label = "Ground Truth", color = "red")
plt.plot(t_lst, predicted_prob_lst, label = "Predicted")
plt.legend()
plt.xlabel("t")
plt.ylabel("P(X > t)")
plt.savefig(f"Plots/{scheme}/tail_prob.png")
plt.clf()
plt.close()

print("Real range:", MIN_REAL, MAX_REAL)
print("Imag range:", MIN_IMAG, MAX_IMAG)

#MIN_REAL, MAX_REAL = float("inf"), -float("inf")
#MIN_IMAG, MAX_IMAG = float("inf"), -float("inf")
#
### Compare probability of X against ground truth
#t_lst = list(range(3, 11))
#true_prob_lst = []
#predicted_prob_lst = []
#for t in tqdm(t_lst):
#    ans = prob(t)
#    true_prob_lst.append(ans)
#    predicted = prob_predicted(mgf_trainer, t)
#    predicted_prob_lst.append(predicted)
#
#plt.scatter(t_lst, true_prob_lst, label = "Ground Truth", color = "red")
#plt.plot(t_lst, predicted_prob_lst, label = "Predicted")
#plt.legend()
#plt.xlabel("t")
#plt.ylabel("P(X = t)")
#plt.savefig(f"Plots/{scheme}/prob.png")
#plt.clf()
#plt.close()
#
#print("Real range:", MIN_REAL, MAX_REAL)
#print("Imag range:", MIN_IMAG, MAX_IMAG)
