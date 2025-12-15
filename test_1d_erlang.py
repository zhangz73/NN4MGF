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
from matplotlib.colors import LogNorm
from tqdm import tqdm

from fit_mgf import *
from inverse_laplace import InverseLaplace

d = 1
TRAIN_LB = -1
TRAIN_UB = 0
TRAIN_IMAG_LB = -30
TRAIN_IMAG_UB = 30
EVAL_LB = -1
EVAL_UB = 0
EVAL_IMAG_LB = -30
EVAL_IMAG_UB = 30
RETRAIN = True

K = 2
LAM = 5

scheme = f"d={d}/erlang"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)

def target_mgf_func(theta):
    mgf = (LAM / (LAM - theta)) ** K
    return mgf

## Training
mgf_trainer = MGFTrainer(d = 1, mu = None, sigma = None, R = None, hidden_dim = 128, dir = f"{scheme}")
if RETRAIN:
    mgf_trainer.train_from_target(target_mgf_func, lb = -1, ub = 0, imag_lb=-0.5, imag_ub=0.5, batch_size = 500, num_epochs = 1000, init_lr = 1e-3, lam_monotone = 0.1, lam_CR = 0.1, lam_growth = 0)
    mgf_trainer.save()
else:
    mgf_trainer.load()

## Compute first moment
first_moment = mgf_trainer.get_first_moment()
print("True mean:", K/LAM)
print("Predicted mean from NN:", first_moment)

## Comparing MGF against ground truth
n_points_per_dim = 10
theta_real_lst = np.linspace(EVAL_LB, -EVAL_UB, n_points_per_dim)
theta_imag_lst = np.linspace(EVAL_IMAG_LB, EVAL_IMAG_UB, n_points_per_dim)
X, Y = np.meshgrid(theta_real_lst, theta_imag_lst)
theta_real_lst, theta_imag_lst = torch.from_numpy(X.ravel()).double(), torch.from_numpy(Y.ravel()).double()
theta_lst = torch.complex(theta_real_lst, theta_imag_lst)

predicted_mgf_lst = mgf_trainer.eval(theta_lst.reshape((-1, 1)))[:,0]
true_mgf_lst = target_mgf_func(theta_lst)
diff_lst = torch.abs(predicted_mgf_lst - true_mgf_lst) ** 2
plt.imshow(
    diff_lst.reshape((n_points_per_dim, n_points_per_dim)),
    extent=[EVAL_LB, -EVAL_UB, EVAL_IMAG_LB, EVAL_IMAG_UB],  # [xmin, xmax, ymin, ymax]
    origin='lower',  # ensures Y-axis goes from bottom to top
    aspect='auto',
    cmap='viridis',
    norm=LogNorm(vmin=diff_lst.min().item() + 1e-12, vmax=diff_lst.max().item())  # log scale
)
print(torch.mean(diff_lst))
plt.colorbar(label='Squared Error')
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.title('MGF Prediction Squared Error')
plt.savefig(f"Plots/{scheme}/mgf_error.png")
plt.clf()
plt.close()
