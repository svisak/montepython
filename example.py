#!/usr/bin/env python

import numpy as np
from montepython.hmc import HMC

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

dim = 2
startpos = np.zeros(dim)
n_samples = 100
ell = 100
epsilon = 0.3

def lnprior(q):
    return 0

# A simple box distribution
def lnlikelihood(q):
    if np.amax(np.abs(q)) < 5:
        return 0
    else:
        return np.NINF

# The gradient of the negative log posterior
def gradient(q):
    return 0

hmc = HMC(gradient=gradient, ell=ell, epsilon=epsilon, dim=dim, startpos=startpos, lnprior=lnprior, lnlikelihood=lnlikelihood)
hmc.set_seed(1234)
hmc.run(n_samples)

print(hmc.acceptance_fraction())
chain = hmc.get_chain()

# Put chain in histogram etc
