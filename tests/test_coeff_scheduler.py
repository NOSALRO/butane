import unittest
import torch
import butane

class TestScheduler(unittest.TestCase):

    def test_coeff(self):
        a_s = butane.optim.CoefficientScheduler(0.3, 0.01, 100, mode='constant')
        for i in range(100):
            print(a_s())
            a_s.update()

if __name__ == '__main__':
    unittest.main()
