#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from rwm import RandomWalkMetropolis
from hmc import HMC

def prior(q):
    if np.amax(np.abs(q)) < 10:
        return 0
    else:
        return np.NINF

def likelihood(q):
    return -0.5 * q**2

def gradient(q):
    return q*np.exp(-0.5 * q**2)

dim = 1
startpos = np.zeros(dim)
stepsize = 2.5

rwm = RandomWalkMetropolis(stepsize, dim, startpos, prior, likelihood)
rwm.run(10000)

plt.figure(figsize=(4.5, 3.0))
x = np.linspace(-3, 3)
y = np.sqrt(1/2/np.pi) * np.exp(-0.5 * x**2)
plt.plot(x, y)

plt.hist(rwm.get_chain(), bins=500, density=True, stacked=True)
plt.savefig('r.pdf', bbox_inches='tight')

hmc = HMC(gradient, 30, 0.001, dim, startpos, prior, likelihood)
hmc.run(10000)

plt.figure(figsize=(4.5, 3.0))
plt.hist(hmc.get_chain(), bins=500, density=True, stacked=True)
plt.plot(x, y)
plt.savefig('h.pdf', bbox_inches='tight')

print(hmc.acceptance_rate())
print(rwm.acceptance_rate())
