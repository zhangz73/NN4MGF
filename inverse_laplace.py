import os
import mpmath
from mpmath import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm

class InverseLaplace:
    def __init__(self, mgf_trainer, dps = 100, pretty = True):
        self.mgf_trainer = mgf_trainer
        mp.dps = dps
        mp.pretty = pretty
        self.d = self.mgf_trainer.d
    
    def eval(self, p, k = 0):
        p = float(p)
        input = torch.zeros((1, self.d)).float()
        input[:,k-1] = -p
        with torch.no_grad():
            output = self.mgf_trainer.eval(input)
        return output[0,k].item()

    def invert(self, t, k = 0):
        t = float(t)
        fp = lambda p: self.eval(p, k = k)
        return float(invertlaplace(fp, t, method = "stehfest"))
