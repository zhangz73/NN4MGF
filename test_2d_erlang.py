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

d = 2
TRAIN_LB = -1
TRAIN_UB = 1
TRAIN_IMAG_LB = -1
TRAIN_IMAG_UB = 1
EVAL_LB = -1
EVAL_UB = 1
EVAL_IMAG_LB = -1
EVAL_IMAG_UB = 1
RETRAIN = True
S_LST = []

K_LST = [2, 3]
LAM_LST = [2, 3]

MIN_REAL, MAX_REAL = float("inf"), -float("inf")
MIN_IMAG, MAX_IMAG = float("inf"), -float("inf")

scheme = f"d={d}/erlang"

os.makedirs(f"Plots/{scheme}", exist_ok=True)
os.makedirs(f"Models/{scheme}", exist_ok=True)
os.makedirs(f"Logs/{scheme}", exist_ok=True)

def target_mgf_func(theta):
    mgf = 1
    for i in range(d):
        LAM = LAM_LST[i]
        K = K_LST[i]
        mgf = mgf * (LAM / (LAM - theta[:,i])) ** K
    return mgf

## Training
mgf_trainer = MGFTrainer(d = d, mu = None, sigma = None, R = None, hidden_dim = 128, dir = f"{scheme}", x_min = TRAIN_LB, x_max = TRAIN_UB, y_min = TRAIN_IMAG_LB, y_max = TRAIN_IMAG_UB)
theta_eval = mgf_trainer.sample_vector(lb=EVAL_LB, ub=EVAL_UB, imag_lb=EVAL_IMAG_LB, imag_ub=EVAL_IMAG_UB, batch_size = 10000)
if RETRAIN:
    mgf_trainer.train_from_target(target_mgf_func, full_gradient = True, theta_eval = theta_eval, lb = TRAIN_LB, ub = TRAIN_UB, imag_lb=TRAIN_IMAG_LB, imag_ub=TRAIN_IMAG_UB, batch_size = 2048, num_epochs = 5000, init_lr = 1e-3, lam_monotone = 0, lam_CR = 1e-1, lam_growth = 0)
    mgf_trainer.save()
else:
    mgf_trainer.load()

## Comparing against ground truth
mgf_true = target_mgf_func(theta_eval)
with torch.no_grad():
    output = mgf_trainer.eval(theta_eval)
    output = output.to(device = theta_eval.device)
    mgf_pred = output[:,0]
mgf_true, mgf_pred = mgf_true.cpu(), mgf_pred.cpu()
theta_eval = theta_eval.cpu()
mgf_trainer.plot_compare(mgf_pred, mgf_true, title = "Interior")
