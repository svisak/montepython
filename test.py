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
startpos = np.zeros(dim)
stepsize = 4.5

rwm = RWM(stepsize, dim, startpos, lnprior, lnlikelihood)
rwm.run(10000)

plt.figure(figsize=(4.5, 3.0))
x = np.linspace(-3, 3)
y = np.sqrt(1/2/np.pi) * np.exp(-0.5 * x**2)
plt.plot(x, y)

plt.hist(rwm.get_chain(), bins=500, density=True, stacked=True)
plt.savefig('fig/rwm.pdf', bbox_inches='tight')

hmc = HMC(gradient, 100, 0.001, dim, startpos, lnprior, lnlikelihood)
hmc.run(50000)

plt.figure(figsize=(4.5, 3.0))
plt.hist(hmc.get_chain(), bins=500, density=True, stacked=True)
plt.plot(x, y)
plt.savefig('fig/hmc.pdf', bbox_inches='tight')

plt.figure(figsize=(4.5, 3.0))
plt.hist(rwm.get_chain(), bins=500, density=True, stacked=True)
plt.hist(hmc.get_chain(), bins=500, density=True, stacked=True)
plt.savefig('fig/both.pdf', bbox_inches='tight')

print('Acceptance rate HMC: {}'.format(hmc.acceptance_rate()))
print('Acceptance rate RWM: {}'.format(rwm.acceptance_rate()))
