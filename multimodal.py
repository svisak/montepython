#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from montepython.hmc import HMC
from montepython.rwm import RWM

# TARGET DISTRIBUTION
dim = 1
mu = 4
var = 1

startpos = 0
n_samples = 5000

# PRIOR, LIKELIHOOD, GRADIENT
def lnprior(q):
    return 0

def lnlikelihood(q):
    val = np.log(right(q) + left(q))
    return val

def left(q):
    return np.exp(-0.5*(q+mu)**2/var)

def right(q):
    return np.exp(-0.5*(q-mu)**2/var)

def gradient(q):
    fprime = -(q-mu)/var * right(q) - (q+mu)/var * left(q)
    f = right(q) + left(q)
    return -fprime/f

def run_hmc(n, gradient, ell, epsilon, dim, startpos, lnprior, lnlikelihood):
    hmc = HMC(gradient, ell, epsilon, dim, startpos, lnprior, lnlikelihood)
    hmc.run(n)
    print('Acceptance rate HMC: {:.10f}'.format(hmc.acceptance_rate()))
    return hmc

def run_rwm(n, stepsize, dim, startpos, lnprior, lnlikelihood):
    rwm = RWM(stepsize, dim, startpos, lnprior, lnlikelihood)
    rwm.run(n)
    print('Acceptance rate RWM: {:.10f}'.format(rwm.acceptance_rate()))
    return rwm

def plot_likelihood(lnlikelihood):
    plt.figure(figsize=(4.5, 3.0))
    q = np.linspace(-8, 8)
    pi = np.exp(lnlikelihood(q))
    plt.plot(q, pi)
    plt.savefig('tmp.pdf', bbox_inches='tight')

def plot(mcmc, L, eps):
    plt.figure(figsize=(8.0, 4.5))
    plt.subplot(1,2,1)
    plt.hist(mcmc.get_chain(), bins=1000, density=True, stacked=True)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.xlim(-8, 8)
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(mcmc.get_chain())), mcmc.get_chain())
    plt.xlabel('Sample number')
    plt.ylabel(r'$q$')
    plt.title('Traceplot')
    plt.savefig('fig/{}_multimodal_traceplot_L{}_eps{}.pdf'.format(mcmc.get_mcmc_type(), L, eps), bbox_inches='tight')

def plot_rwm(mcmc, stepsize):
    plt.figure(figsize=(8.0, 4.5))
    plt.subplot(1,2,1)
    plt.hist(mcmc.get_chain(), bins=1000, density=True, stacked=True)
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\pi(q)$')
    plt.xlim(-8, 8)
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(mcmc.get_chain())), mcmc.get_chain())
    plt.xlabel('Sample number')
    plt.ylabel(r'$q$')
    plt.title('Traceplot')
    plt.savefig('fig/{}_multimodal_traceplot_stepsize{}.pdf'.format(mcmc.get_mcmc_type(), stepsize), bbox_inches='tight')
    
plot_likelihood(lnlikelihood)
L = 50
eps = 0.1
#hmc = run_hmc(n_samples, gradient, L, eps, dim, startpos, lnprior, lnlikelihood)
#plot(hmc, L, eps)
stepsize = 7.0
rwm = run_rwm(n_samples, stepsize, dim, startpos, lnprior, lnlikelihood)
plot_rwm(rwm, stepsize)