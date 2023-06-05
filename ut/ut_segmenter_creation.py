import unittest
import numpy as np
from preprocessing.segmentation import segment_data, gauss_conv
from preprocessing.creation import slices_to_npy, make_raw_data
from utils.Types import SNR, Density, Particle
from utils.definitions import (
    SIZE,
    DEFAULT_SLICES_PATH,
    DTS_RAW_PATH,
    DEFAULT_DATA_PATH
)


class TestSegmenter(unittest.TestCase):
    def test_gauss_conv(self):
        img_3d = np.zeros(shape=(3, 3, 3))  # (H,W,D)
        particles = [Particle(0, 1.0, 2.0, 2.0), Particle(2, 1.5, 0.8, 0.2)]
        filtered_img = gauss_conv(img_3d, particles, kernel=3, sigma=0.5)
        self.assertTrue(filtered_img is not None)
        self.assertEqual(filtered_img.shape, (3, 3, 3))

    def test_segment_data(self):
        img_3d, particles = segment_data(SNR.TYPE_7, Density.LOW, t=5, kernel=3, sigma=0.5)
        self.assertTrue(img_3d is not None)
        self.assertTrue(len(particles) > 0)
        self.assertEqual(img_3d.shape, (SIZE[0], SIZE[1], SIZE[2]))


class TestCreation(unittest.TestCase):
    def test_slices_to_npy(self):
        slices_to_npy(SNR.TYPE_7, Density.MID, DEFAULT_SLICES_PATH, DEFAULT_DATA_PATH)

    def test_make_raw_data(self):
        make_raw_data(SNR.TYPE_7, Density.HIGH, kernel=3, sigma=0.5, dest_dir=DTS_RAW_PATH)


if __name__ == '__main__':
    unittest.main()
