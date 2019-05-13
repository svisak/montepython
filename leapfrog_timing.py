#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from montepython.hmc import HMC
from montepython.rwm import RWM
from montepython import utils

def get_distributions(dim):
    mu = 1 * np.ones(dim)
    var = 0.8 * np.eye(dim)
    var_inv = np.linalg.inv(var)
    
    def lnprior(q):
        return 0
    
    def lnlikelihood(q):
        tmp = np.matmul(var_inv, q-mu)
        return -0.5 * np.matmul(q-mu, tmp)

    def gradient(q):
        return np.matmul(var_inv, (q-mu))

    return (lnprior, lnlikelihood, gradient)


dims = np.zeros(100)
for i in range(100):
    dims[i] = i+1
dims = dims.astype(int)

ells = np.ones(100).astype(int)
for i in range(100):
    ells[i] = 5 * (i+1)
ells = ells.astype(int)

time_dims = np.empty(len(dims))
time_ells = np.empty(len(ells))

# Time vs dim
for i in range(len(dims)):
    d = dims[i] 
    lnprior, lnlikelihood, gradient = get_distributions(d)
    hmc = HMC(gradient, 100, 0.1, d, np.ones(d), lnprior, lnlikelihood)
    hmc.run(50)
    time_dims[i] = hmc.exec_time

# Time vs ell
for i in range(len(ells)):
    d = 16
    lnprior, lnlikelihood, gradient = get_distributions(d)
    hmc = HMC(gradient, ells[i], 0.1, d, np.ones(d), lnprior, lnlikelihood)
    hmc.run(50)
    time_ells[i] = hmc.exec_time

plt.figure(figsize=(8, 4.5))

plt.subplot(1, 2, 1)
plt.plot(dims, time_dims, label=r'$L = 100$')
plt.xlabel('Dimensions')
plt.ylabel('Time / s')
plt.title('Leapfrog execution time vs number of dimensions')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ells, time_ells, label=r'Dim $= 16$')
plt.xlabel(r'$L$')
plt.ylabel(r'Time / s')
plt.title('Leapfrog execution time vs number of steps L')
plt.legend()

plt.tight_layout()
plt.savefig('fig/leapfrog_timing.pdf', bbox_inches='tight')
