#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time

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
        print('=========== {}D ==========='.format(d))
        startpos = 2 * np.ones(d)
        gauss = Gauss(d)
        print('Running HMC chain')
        t = time.time()
        hmc = HMC(gauss.gradient, ell, epsilon, d, startpos, gauss.lnprior, gauss.lnlikelihood)
        hmc.run(n_samples)
        print('HMC chain finished, time = {} s'.format(time.time() - t))
        acc_rate.append(hmc.acceptance_rate())
        acors.append(autocorrelation(hmc.get_chain(), max_lag))
    return (acc_rate, acors)

def run_rwm(n_samples, dims, max_lag):
    acc_rate = []
    acors = []
    stepsizes = [3.3891544377720013, 1.1817250826203343, 0.4578245093723353, 0.24330671508534327, 0.10473556281705265, 0.05566077223905729, 0.026622374813547164, 0.01572024610365147, 0.009282648121745157]
    for i in range(len(dims)):
        d = dims[i]
        print('=========== {}D ==========='.format(d))
        print('dim =', d)
        startpos = 2 * np.ones(d)
        gauss = Gauss(d)
        acc = 0.
        toohigh = 0
        toolow = 0
        print('Running RWM chain')
        t = time.time()
        while acc > 0.25 or acc < 0.20:
            if 2 < toohigh and 2 < toolow:
                print('Infinite loop')
                exit(1)
            rwm = RWM(stepsizes[i], d, startpos, gauss.lnprior, gauss.lnlikelihood)
            rwm.run(n_samples)
            acc = rwm.acceptance_rate() 
            if acc > 0.25:
                stepsizes[i] *= 1.1
                toohigh += 1
            elif acc < 0.20:
                stepsizes[i] *= 0.9
                toolow += 1
        print('RWM chain finished, time = {} s'.format(time.time() - t))
        print('Retries =', toohigh+toolow)
        acors.append(autocorrelation(rwm.get_chain(), max_lag))
        acc_rate.append(acc)
    return (stepsizes, acc_rate, acors)
    

n_samples = 10000
max_lag = min(n_samples//10, 200)
D = np.arange(1,10)
dims = 2 ** D
stepsize_rwm, acc_rate_rwm, acors_rwm = run_rwm(n_samples, dims, max_lag)
print('stepsize_rwm = ', stepsize_rwm)
print('Random walk finished')
acc_rate_hmc, acors_hmc = run_hmc(n_samples, dims, max_lag)

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

plt.subplot(2, 3, 2)
for i in range(len(dims)):
    # This should probably plot the most correlated rather than just the last dimensions
    plt.plot(acors_rwm[i][:, i], label=r'dim = {}'.format(dims[i]))
plt.xlabel(r'Lag')
plt.ylabel(r'Autocorrelation')
plt.title(r'Autocorrelation RWM')
plt.legend()

plt.subplot(2, 3, 5)
for i in range(len(dims)):
    # This should probably plot the most correlated rather than just the last dimensions
    plt.plot(acors_hmc[i][:, i], label=r'dim = {}'.format(dims[i]))
plt.xlabel(r'Lag')
plt.ylabel(r'Autocorrelation')
plt.title(r'Autocorrelation HMC')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(dims, stepsize_rwm)
plt.xlabel(r'Dimensions')
plt.ylabel(r'$\sigma$')
plt.title(r'$\sigma$ of RWM proposal distribution, $\mathcal{N}\left(0, \sigma^2\right)$')

plt.tight_layout()
plt.savefig('fig/rwm_hmc_correlation_comparison.pdf')
