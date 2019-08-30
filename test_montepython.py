#!/usr/bin/env python3

import unittest
import numpy as np
from montepython.mcmc import MetaChain
from montepython.hmc import HMC, State
from montepython.rwm import RWM
from montepython import utils

class ChainTestCase(unittest.TestCase):

    def test_chain_init(self):
        for i in range(1, 4):
            with self.subTest(i=i):
                metachain = MetaChain(i, self.randpos(i))
                self.assertEqual(metachain._index, 0)
                self.assertEqual(metachain.chain().shape, (1,i))
                self.assertEqual(metachain.acceptance_fraction(), 1)

    def test_accept_reject(self):
        for i in range(1, 6):
            with self.subTest(i=i):
                q = self.randpos(i)
                metachain = MetaChain(i, q)
                self.assertEqual(metachain.acceptance_fraction(), 1)
                metachain.extend(10)
                metachain.reject()
                self.assertEqual(metachain.acceptance_fraction(), 0.5)
                self.assertTrue((metachain.head() == q).all())
                q = self.randpos(i)
                metachain.accept(q)
                self.assertTrue((metachain.head() == q).all())

    def randpos(self, ndim):
        return np.random.multivariate_normal(np.zeros(ndim), np.eye(ndim))

class MCMCTestCase(unittest.TestCase):

    def test_lnposterior(self):
        def lnprior(x):
            return x
        def lnlikelihood(x):
            return x
        rwm = RWM(stepsize=1, dim=1, startpos=1, lnprior=lnprior, lnlikelihood=lnlikelihood)
        self.assertEqual(rwm.lnposterior(2), 4)
        self.assertEqual(rwm.lnposterior(np.NINF), np.NINF)
        with self.assertRaises(ValueError):
            rwm.lnposterior(np.nan)

class HMCTestCase(unittest.TestCase):

    def setUp(self):
        def lnprior(position):
            return 1

        def lnlikelihood(position):
            return 1

        def gradient(position):
            return np.ones(self.dim)

        self.dim = 10
        startpos = np.zeros(self.dim)
        ell = 1
        epsilon = 1.0
        self.hmc = HMC(gradient=gradient, leapfrog_ell=ell, leapfrog_epsilon=epsilon, dim=self.dim, startpos=startpos, lnprior=lnprior, lnlikelihood=lnlikelihood)

    def test_chain_size(self):
        self.assertEqual(len(self.hmc.chain()), 1)
        self.assertEqual(self.hmc.chain().shape, (1,10))
        self.hmc.run(50)
        self.assertEqual(len(self.hmc.chain()), 51)
        self.assertEqual(self.hmc.chain().shape, (51,10))
        self.assertEqual(self.hmc._metachain._index, 50)

    def test_potential(self):
        self.assertEqual(self.hmc.potential(0), -2)

    def test_hamiltonian(self):
        position = 0
        momentum = 2*np.ones(self.dim)
        potential = self.hmc.potential(position)
        kinetic = self.hmc.kinetic(momentum)
        self.assertEqual(self.hmc.hamiltonian(potential, kinetic), 18)

class RWMTestCase(unittest.TestCase):

    def setUp(self):
        def lnprior(position):
            return 1

        def lnlikelihood(position):
            return 1

        self.dim = 2
        startpos = np.zeros(self.dim)
        stepsize = 4.0
        self.rwm = RWM(stepsize=stepsize, dim=self.dim, startpos=startpos, lnprior=lnprior, lnlikelihood=lnlikelihood)

    def test_chain_size(self):
        self.assertEqual(len(self.rwm.chain()), 1)
        self.assertEqual(self.rwm.chain().shape, (1,2))
        self.rwm.run(25)
        self.assertEqual(len(self.rwm.chain()), 26)
        self.assertEqual(self.rwm.chain().shape, (26,2))
        self.assertEqual(self.rwm._metachain._index, 25)

class UtilsTestCase(unittest.TestCase):

    def test_autocorrelation_small(self):
        chain = np.empty((10, 1))
        for i in range(10, 0, -1):
            chain[i-1, 0] = i
        max_lag = 1
        acors = utils.autocorrelation(chain, max_lag)
        self.assertEqual(acors[0, 0], 1)
        self.assertLess(acors[max_lag, 0]-0.927710843373494, 0.00001)

    '''
    This test can theoretically fail even when the autocorrelation method
    is implemented correctly, since the test relies on random numbers being
    sufficiently uncorrelated.
    '''
    def test_autocorrelation_big(self):
        tol = 1e-5
        max_dimensions = 10
        max_lag = 100
        for dim in range(1, max_dimensions+1):
            chain = np.random.rand(10000, dim)
            acors = utils.autocorrelation(chain, max_lag)
            self.assertLess(np.abs(np.amax(acors[0, :]))-1, tol)
            self.assertLess(np.abs(np.amin(acors[0, :]))-1, tol)
            self.assertLess(np.amax(acors[1:, :]), 0.05)

class BatchTestCase(unittest.TestCase):

    def setUp(self):
        def lnprior(position):
            return 1

        def lnlikelihood(position):
            return 1

        def gradient(position):
            return np.ones(self.dim)

        self.dim = 10
        startpos = np.zeros(self.dim)
        ell = 1
        epsilon = 1.0
        self.hmc = HMC(gradient=gradient, leapfrog_ell=ell, leapfrog_epsilon=epsilon, dim=self.dim, startpos=startpos, lnprior=lnprior, lnlikelihood=lnlikelihood)

    def test_chain_size(self):
        self.assertEqual(len(self.hmc.chain()), 1)
        self.assertEqual(self.hmc.chain().shape, (1,10))
        self.hmc.run(n_samples=52, batch_size=10)
        self.assertEqual(len(self.hmc.chain()), 53)
        self.assertEqual(self.hmc.chain().shape, (53,10))
        self.assertEqual(self.hmc._metachain._index, 52)

if __name__ == '__main__':
    unittest.main()
