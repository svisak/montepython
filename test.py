#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rwm import RWM
from hmc import HMC

# Specify distributions
def lnprior(q):
    return 0 # Prior is just 1

def lnlikelihood(q):
    return -0.5 * q**2

def gradient(q):
    #return -np.sqrt(1/2/np.pi) * q * np.exp(-0.5 * q**2)
    return -q

# Run parameters
dim = 1
startpos = 3 * np.ones(dim)

# A normal distribution
x = np.linspace(-3, 3)
y = np.sqrt(1/2/np.pi) * np.exp(-0.5 * x**2)

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
    hmc = HMC(gradient, 500, 0.001, dim, startpos, lnprior, lnlikelihood)
    mi = 1.0
    hmc.scale_mass_matrix(mi)
    hmc.run(200000)
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

hmc_test()
