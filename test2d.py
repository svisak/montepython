#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from hmc import HMC
from rwm import RWM
import utils

# TARGET DISTRIBUTION
dim = 16
mu = 4 * np.ones(dim)
var = np.eye(dim)
var_inv = np.linalg.inv(var)

# COMMON RUN PARAMETERS
startpos = np.zeros(dim)
n_samples = 50000

# PRIOR, LIKELIHOOD, GRADIENT
def lnprior(q):
    return 0

def lnlikelihood(q):
    tmp = np.matmul(var_inv, (q-mu))
    return -0.5 * np.matmul((q-mu), tmp)

def gradient(q):
    return np.matmul(var_inv, (q-mu))

# HMC
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

def heatmap2d(mcmc):
    q1 = mcmc.get_chain()[:, 0]
    q2 = mcmc.get_chain()[:, 1]
    plt.figure(figsize=(4.5, 4.5))
    plt.hist2d(q1, q2, bins=40)
    plt.savefig('tmp.pdf')

def surface2d(mcmc):
    x = hmc.get_chain()[:, 0]
    y = hmc.get_chain()[:, 1]
    fig = plt.figure(figsize=(4.5, 3.0))
    ax = fig.gca(projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=(60,60))

    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title('Title')
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.savefig('tmp.pdf')
    
def time_leapfrog():
    def lnpr(q):
        return 0

    def lnlik(q):
        tmp = np.matmul(var_inv, (q-mu))
        return -0.5 * np.matmul((q-mu), tmp)

    def grad(q, ):
        return np.matmul(var_inv, (q-mu))

    dims = np.zeros(100)
    for i in range(100):
        dims[i] = i+1
    dims = dims.astype(int)
    ells = np.ones(100).astype(int)
    for i in range(100):
        ells[i] = 5 * (i+1)
    time_dims = np.empty(len(dims))
    time_ells = np.empty(len(ells))
    for i in range(len(dims)):
        d = dims[i] 
        m = 4 * np.ones(d)
        v = np.eye(d)
        v_inv = np.linalg.inv(v)
        def lnpr(q):
            return 0
        
        def lnlik(q):
            tmp = np.matmul(v_inv, (q-m))
            return -0.5 * np.matmul((q-m), tmp)
        
        def grad(q):
            return np.matmul(v_inv, (q-m))
        hmc = HMC(grad, 100, 0.1, d, np.ones(d), lnpr, lnlik)
        hmc.run(50)
        time_dims[i] = hmc.exec_time
    for i in range(len(ells)):
        d = 16
        hmc = HMC(gradient, ells[i], 0.1, d, np.ones(d), lnprior, lnlikelihood)
        hmc.run(50)
        time_ells[i] = hmc.exec_time

    plt.figure(figsize=(4.5, 3.0))
    plt.plot(dims, time_dims)
    plt.xlabel('Dimensions / pcs')
    plt.ylabel('Time / s')
    plt.title('Leapfrog execution time vs number of dimensions, 50 samples, L = 100')
    plt.savefig('time_vs_dim.pdf', bbox_inches='tight')

    plt.figure(figsize=(4.5, 3.0))
    plt.plot(ells, time_ells)
    plt.xlabel('L / pcs')
    plt.ylabel('Time / s')
    plt.title('Leapfrog execution time vs number of steps L, 50 samples, 16 dimensions')
    plt.savefig('time_vs_ell.pdf', bbox_inches='tight')

time_leapfrog()
#hmc = run_hmc(n_samples, gradient, 100, 0.1, dim, startpos, lnprior, lnlikelihood)
#rwm = run_rwm(n_samples, 7.0, dim, startpos, lnprior, lnlikelihood)
#heatmap2d(hmc)
#surface2d(hmc)
