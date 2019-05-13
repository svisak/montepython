#!/usr/bin/env python

import numpy as np
from montepython.hmc import HMC

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

dim = 2
startpos = np.zeros(dim)
n_samples = 100000
ell = 100
epsilon = 0.3

def lnprior(q):
    return 0

def lnlikelihood(q):
    if np.amax(np.abs(q)) < 5:
        return 0
    else:
        return np.NINF

def gradient(q):
    return 0

hmc = HMC(gradient, ell, epsilon, dim, startpos, lnprior, lnlikelihood)
hmc.run(n_samples)
