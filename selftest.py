import unittest
import numpy as np
from montepython import Chain
from hmc import HMC
import utils

class ChainTestCase(unittest.TestCase):

    def test_chain_init(self):
        for i in range(1, 4):
            with self.subTest(i=i):
                chain = Chain(i)
                self.assertEqual(chain.current_index(), -1)
                self.assertEqual(chain.get_chain().shape, (0,i))
                with self.assertRaises(ZeroDivisionError):
                    chain.acceptance_rate()

    def test_accept_reject(self):
        for i in range(1, 6):
            with self.subTest(i=i):
                chain = Chain(i)
                chain.extend(10)
                q = np.random.multivariate_normal(np.zeros(i), np.eye(i))
                chain.accept(q)
                self.assertEqual(chain.acceptance_rate(), 1)
                self.assertTrue((chain.head() == q).all())
                chain.reject()
                self.assertEqual(chain.acceptance_rate(), 0.5)
                self.assertTrue((chain.head() == q).all())

class HMCTestCase(unittest.TestCase):

    def setUp(self):
        def lnprior(position):
            return 1

        def lnlikelihood(position):
            return 1

        def gradient(position):
            return 0

        self.dim = 10
        startpos = np.zeros(self.dim)
        self.hmc = HMC(gradient, 1, 1, self.dim, startpos, lnprior, lnlikelihood)

    def test_hamiltonian(self):
        position = 0
        momentum = 2*np.ones(self.dim)
        self.assertEqual(self.hmc.hamiltonian(position, momentum), 18)

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

if __name__ == '__main__':
    unittest.main()
