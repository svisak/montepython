#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from montepython.hmc import HMC
from montepython.rwm import RWM
from montepython.utils import autocorrelation, autocorrelation_time

class Gauss():

    def __init__(self, dim):
        self.dim = dim
        self.mu = 3 * np.ones(dim)
        var = 0.5 * np.eye(dim)
        self.var_inv = np.linalg.inv(var)

    def lnlikelihood(self, q):
        tmp = np.matmul(self.var_inv, q-self.mu)
        return -0.5 * np.matmul(q-self.mu, tmp)

    def lnprior(self, q):
        return 0

    def gradient(self, q):
        return np.matmul(self.var_inv, q-self.mu)

def run_hmc(n_samples, dims, max_lag):
    ell = 100
    epsilon = 0.1
    acc_rate = []
    acors = []
    for d in dims:
        startpos = 2 * np.ones(d)
        gauss = Gauss(d)
        hmc = HMC(gauss.gradient, ell, epsilon, d, startpos, gauss.lnprior, gauss.lnlikelihood)
        hmc.run(n_samples)
        acc_rate.append(hmc.acceptance_rate())
        acors.append(autocorrelation(hmc.get_chain(), max_lag))
    return (acc_rate, acors)

def run_rwm(n_samples, dims, max_lag):
    acc_rate = []
    acors = []
    stepsizes = []
    for d in dims:
        startpos = 2 * np.ones(d)
        gauss = Gauss(d)
        stepsize = 12.0 # Starting point for stepsize
        acc = 0.
        while acc > 0.25 or acc < 0.20:
            rwm = RWM(stepsize, d, startpos, gauss.lnprior, gauss.lnlikelihood)
            rwm.run(n_samples)
            acc = rwm.acceptance_rate() 
            if acc > 0.25:
                stepsize *= 1.1
            elif acc < 0.20:
                stepsize *= 0.9
            else:
                acors.append(autocorrelation(rwm.get_chain(), max_lag))
        stepsizes.append(stepsize)
        acc_rate.append(acc)
    return (stepsizes, acc_rate, acors)
    

n_samples = 10000
max_lag = min(n_samples//10, 200)
dims = np.arange(1,21)
acc_rate_hmc, acors_hmc = run_hmc(n_samples, dims, max_lag)
stepsize_rwm, acc_rate_rwm, acors_rwm = run_rwm(n_samples, dims, max_lag)

np.save('data/acc_rate_hmc.npy', acc_rate_hmc)
np.save('data/acc_rate_rwm.npy', acc_rate_rwm)
for d in dims:
    i = d - 1
    np.save('data/acors_hmc_{}d.npy'.format(d), acors_hmc[i])
    np.save('data/acors_rwm_{}d.npy'.format(d), acors_rwm[i])
np.save('data/stepsize_rwm.npy', stepsize_rwm)

# Plot
plt.figure(figsize=(16.0, 9.0))

plt.subplot(2, 3, 1)
plt.plot(dims, acc_rate_rwm)
plt.xlabel(r'Dimensions')
plt.ylabel(r'Acceptance rate')
plt.ylim(0, 1)
plt.title(r'Acceptance rate RWM')

plt.subplot(2, 3, 4)
plt.plot(dims, acc_rate_hmc)
plt.xlabel(r'Dimensions')
plt.ylabel(r'Acceptance rate')
plt.ylim(0, 1)
plt.title(r'Acceptance rate HMC')

plot_d = [1, 10, 20]
plt.subplot(2, 3, 2)
for d in plot_d:
    i = d - 1
    plt.plot(acors_rwm[i][:, i], label=r'dim = {}'.format(d))
plt.xlabel(r'Lag')
plt.ylabel(r'Autocorrelation')
plt.title(r'Autocorrelation RWM')
plt.legend()

plt.subplot(2, 3, 5)
for d in plot_d:
    i = d - 1
    plt.plot(acors_hmc[i][:, i], label=r'dim = {}'.format(d))
plt.xlabel(r'Lag')
plt.ylabel(r'Autocorrelation')
plt.title(r'Autocorrelation HMC')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(dims, stepsize_rwm)
plt.xlabel(r'Dimensions')
plt.ylabel(r'$\sigma$')
plt.title(r'$\sigma$ of RWM proposal distribution')

plt.tight_layout()
plt.savefig('tmp.pdf')
