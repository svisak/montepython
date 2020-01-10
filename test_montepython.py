#!/usr/bin/env python3

import unittest
import numpy as np

from montepython.bayes import Bayes
from montepython.hmc import HMC
from montepython.rwm import RWM
from montepython.state import State
from montepython.metachain import MetaChain
from montepython import diagnostics


class ChainTestCase(unittest.TestCase):

    def test_chain_init(self):
        for i in range(1, 4):
            with self.subTest(i=i):
                state = State(position=self.randpos(i))
                metachain = MetaChain(state)
                self.assertEqual(metachain.chain_with_startpos().shape, (1,i))
                self.assertEqual(metachain.chain().shape, (0,i))
                self.assertEqual(metachain.acceptance_rate(), 1)

    def test_accept_reject(self):
        for i in range(1, 6):
            with self.subTest(i=i):
                q = self.randpos(i)
                state = State(position=q)
                metachain = MetaChain(state)
                self.assertEqual(metachain.acceptance_rate(), 1)
                metachain.reject()
                self.assertEqual(metachain.acceptance_rate(), 0.5)
                self.assertTrue((metachain.head().get('position') == q).all())
                q = self.randpos(i)
                state = State(position=q)
                metachain.accept(state)
                self.assertTrue((metachain.head().get('position') == q).all())

    def randpos(self, ndim):
        return np.random.multivariate_normal(np.zeros(ndim), np.eye(ndim))

class HMCTestCase(unittest.TestCase):

    class Bayes(Bayes):
    
        def __init__(self):
            super().__init__()
    
        def evaluate(self, position):
            self.set_lnlikelihood_value(1)
            self.set_lnprior_value(1)
            self.set_nlp_gradient_value(np.ones(10))

    def setUp(self):
        bayes = self.Bayes()
        startpos = np.zeros(10)
        ell = 1
        epsilon = 1.0
        self.hmc = HMC(bayes, startpos, leapfrog_ell=ell, leapfrog_epsilon=epsilon)

    def test_chain_size(self):
        self.assertEqual(len(self.hmc.chain()), 0)
        self.assertEqual(self.hmc.chain().shape, (0,10))
        self.hmc.run(50)
        self.assertEqual(len(self.hmc.chain()), 50)
        self.assertEqual(self.hmc.chain().shape, (50,10))

    def test_potential(self):
        dummy_state = State(position=0, lnposterior=2)
        self.assertEqual(self.hmc.potential(dummy_state), -2)

    def test_hamiltonian(self):
        dummy_state = State(position=0, momentum=2*np.ones(10), lnposterior=2)
        self.assertEqual(self.hmc.hamiltonian(dummy_state), 18)

class RWMTestCase(unittest.TestCase):

    class Bayes(Bayes):
    
        def __init__(self):
            super().__init__()
    
        def evaluate(self, position):
            self.set_lnlikelihood_value(1)
            self.set_lnprior_value(1)

    def setUp(self):
        bayes = self.Bayes()

        startpos = np.zeros(2)
        stepsize = 4.0
        self.rwm = RWM(bayes, startpos, stepsize=stepsize)

    def test_chain_size(self):
        self.assertEqual(len(self.rwm.chain()), 0)
        self.assertEqual(self.rwm.chain().shape, (0,2))
        self.rwm.run(25)
        self.assertEqual(len(self.rwm.chain()), 25)
        self.assertEqual(self.rwm.chain().shape, (25,2))
        self.assertEqual(len(self.rwm.chain(warmup=10)), 15)

class DiagnosticsTestCase(unittest.TestCase):

    def test_autocorrelation_small(self):
        chain = np.empty((10, 1))
        for i in range(10, 0, -1):
            chain[i-1, 0] = i
        max_lag = 1
        acors = diagnostics.autocorrelation(chain, max_lag)
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
            acors = diagnostics.autocorrelation(chain, max_lag)
            self.assertLess(np.abs(np.amax(acors[0, :]))-1, tol)
            self.assertLess(np.abs(np.amin(acors[0, :]))-1, tol)
            self.assertLess(np.amax(acors[1:, :]), 0.05)

    def test_relative_error(self):
        chain = np.array([[1, 4], [2, 5], [3, 6]])
        rel_err = diagnostics.relative_error(chain)
        correct = np.sqrt(1/3) * np.array([1/2, 1/5])
        self.assertTrue((np.abs(rel_err-correct) < 0.00001).all())

class BatchTestCase(unittest.TestCase):

    class Bayes(Bayes):
    
        def __init__(self):
            super().__init__()
    
        def evaluate(self, position):
            self.set_lnlikelihood_value(1)
            self.set_lnprior_value(1)
            self.set_nlp_gradient_value(np.ones(10))

    def setUp(self):
        bayes = self.Bayes()
        startpos = np.zeros(10)
        ell = 1
        epsilon = 1.0
        self.hmc = HMC(bayes, startpos, leapfrog_ell=ell, leapfrog_epsilon=epsilon)

    def test_chain_size(self):
        self.assertEqual(len(self.hmc.chain_with_startpos()), 1)
        self.assertEqual(len(self.hmc.chain()), 0)
        self.assertEqual(self.hmc.chain_with_startpos().shape, (1,10))
        self.assertEqual(self.hmc.chain().shape, (0,10))
        for n in [10, 10, 10, 10, 10, 2]:
            self.hmc.run(n)
        self.assertEqual(len(self.hmc.chain_with_startpos()), 53)
        self.assertEqual(len(self.hmc.chain()), 52)
        self.assertEqual(self.hmc.chain().shape, (52,10))

if __name__ == '__main__':
    unittest.main()
