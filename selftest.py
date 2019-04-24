import unittest
import numpy as np
from montepython import Chain

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

if __name__ == '__main__':
    unittest.main()
