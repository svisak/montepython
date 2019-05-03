#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rwm import RWM
from hmc import HMC
from utils import autocorrelation

var = 0.8
mu = 4
N = 10000
max_lag = 200

# Specify distributions
def lnprior(q):
    if q > mu+4 or q < mu-4 :
        return np.NINF
    else:
        return 0

def lnlikelihood(q):
    return -0.5 * (q-mu)**2 / var

def gradient(q):
    return (q-mu) / var

# Run parameters
dim = 1
startpos = 0.1 * np.ones(dim)

# A normal distribution
x = np.linspace(mu-4, mu+4)
y = np.sqrt(1/2/np.pi/var) * np.exp(-0.5 * (x-mu)**2/var)

def rwm_test():
    sigma = 3.0
    rwm = RWM(sigma, dim, startpos, lnprior, lnlikelihood)
    rwm.run(N)
    print('Acceptance rate RWM: {}'.format(rwm.acceptance_rate()))

    plt.figure(figsize=(4.5, 3.0))
    plt.plot(x, y)

    plt.hist(rwm.get_chain(), bins=300, density=True, stacked=True)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.title(r'RWM, sigma $= {}$, acceptance\_rate $= {}$'.format(sigma, rwm.acceptance_rate()))
    plt.savefig('fig/rwm_sigma{}_N{}.pdf'.format(sigma, N), bbox_inches='tight')

    plt.figure(figsize=(4.5, 3.0))
    lags = np.arange(0, max_lag+1)
    plt.plot(lags, autocorrelation(rwm.get_chain(), max_lag))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(r'Autocorrelation RWM, $\sigma = {}$, acceptance\_rate $= {}$'.format(sigma, rwm.acceptance_rate()))
    plt.savefig('fig/autocorr_rwm_sigma{}_N{}.pdf'.format(sigma, N), bbox_inches='tight')

def hmc_test():
    ell = 200
    epsilon = 0.3
    #mass_matrix = 1.0*np.eye(dim)
    #hmc = HMC(gradient, ell, epsilon, dim, startpos, lnprior, lnlikelihood, mass_matrix=mass_matrix)
    hmc = HMC(gradient, ell, epsilon, dim, startpos, lnprior, lnlikelihood)
    hmc.run(N)
    print('Acceptance rate HMC: {}'.format(hmc.acceptance_rate()))

    plt.figure(figsize=(4.5, 3.0))
    plt.hist(hmc.get_chain(), bins=300, density=True, stacked=True)
    plt.plot(x, y)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.title(r'HMC, acc\_rate $= {}, L = {}, \epsilon = {}$'.format(hmc.acceptance_rate(), ell, epsilon))
    plt.savefig('fig/hmc_L{}_eps{}_N{}.pdf'.format(ell, epsilon, N), bbox_inches='tight')

    plt.figure(figsize=(4.5, 3.0))
    lags = np.arange(0, max_lag+1)
    plt.plot(lags, autocorrelation(hmc.get_chain(), max_lag))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(r'Autocorrelation HMC, acc\_rate $= {}, L = {}, \epsilon = {}$'.format(hmc.acceptance_rate(), ell, epsilon))
    plt.savefig('fig/autocorr_hmc_L{}_eps{}_N{}.pdf'.format(ell, epsilon, N), bbox_inches='tight')

#rwm_test()
hmc_test()
