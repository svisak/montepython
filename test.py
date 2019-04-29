#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rwm import RWM
from hmc import HMC

var = 4.2
mu = 3

# Specify distributions
def lnprior(q):
    if np.abs(q) > 8:
        return np.NINF
    else:
        return 0

def lnlikelihood(q):
    return -0.5 * (q-mu)**2 / var
    #return 1

def gradient(q):
    return (q-mu) / var
    #return 0 #np.NINF

# Run parameters
dim = 1
startpos = 0.1 * np.ones(dim)

# A normal distribution
x = np.linspace(-4, 8)
y = np.sqrt(1/2/np.pi/var) * np.exp(-0.5 * (x-mu)**2/var)

def rwm_test():
    stepsize = 4.5
    rwm = RWM(stepsize, dim, startpos, lnprior, lnlikelihood)
    rwm.run(10000)
    print('Acceptance rate RWM: {}'.format(rwm.acceptance_rate()))

    plt.figure(figsize=(4.5, 3.0))
    plt.plot(x, y)

    plt.hist(rwm.get_chain(), bins=500, density=True, stacked=True)
    plt.savefig('fig/rwm.pdf', bbox_inches='tight')

def hmc_test():
    hmc = HMC(gradient, 200, 0.2, dim, startpos, lnprior, lnlikelihood)
    var = 1.0
    mi = 1/var/var
    hmc.scale_mass_matrix(mi)
    hmc.run(20000)
    print('Acceptance rate HMC: {}'.format(hmc.acceptance_rate()))

    plt.figure(figsize=(4.5, 3.0))
    plt.hist(hmc.get_chain(), bins=250, density=True, stacked=True)
    plt.plot(x, y)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.title(r'HMC, $m_i = {}$, acceptance\_rate $= {}$'.format(mi, hmc.acceptance_rate()))
    plt.savefig('fig/hmc.pdf', bbox_inches='tight')

def both_test():
    rwm_test()
    hmc_test()
    plt.figure(figsize=(4.5, 3.0))
    plt.hist(rwm.get_chain(), bins=500, density=True, stacked=True)
    plt.hist(hmc.get_chain(), bins=500, density=True, stacked=True)
    plt.savefig('fig/both.pdf', bbox_inches='tight')

def hmc_test_tmp():
    hmc = HMC(gradient, 200, 0.2, dim, startpos, lnprior, lnlikelihood)
    mi = 1.0
    hmc.scale_mass_matrix(mi)
    hmc.run(100000)
    print('Acceptance rate HMC: {}'.format(hmc.acceptance_rate()))

    plt.figure(figsize=(4.5, 3.0))
    plt.hist(hmc.get_chain(), bins=300, density=True, stacked=True)
    plt.plot(x, y)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.title(r'HMC, $m_i = {}$, acceptance\_rate $= {}$'.format(mi, hmc.acceptance_rate()))
    plt.savefig('fig/hmc_mi{}.pdf'.format(mi), bbox_inches='tight')

hmc_test_tmp()
