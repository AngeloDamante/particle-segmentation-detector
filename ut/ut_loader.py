import unittest
from preprocessing.loader import get_slice
from preprocessing.Particle import Particle, SNR, Density


class LoaderUT(unittest.TestCase):
    def test_get_slice(self):
        for t in range(100):
            for depth in range(10):
                idx = (SNR.TYPE_7, Density.LOW, t, depth)
                img = get_slice(*idx)
                self.assertTrue(img)

    # def test_get_3d_image(self):
    #     pass

if __name__ == '__main__':
    unittest.main()
