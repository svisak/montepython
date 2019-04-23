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
rwm.run(20000)

plt.hist(rwm.get_chain(), bins=500)
#plt.savefig('tmp.pdf', bbox_inches='tight')

hmc = HMC(gradient, 30, 0.001, dim, startpos, prior, likelihood)
hmc.run(20000)

plt.hist(hmc.get_chain(), bins=500)
plt.savefig('tmp.pdf', bbox_inches='tight')

print(hmc.acceptance_rate())
print(rwm.acceptance_rate())
