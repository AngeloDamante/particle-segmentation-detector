import unittest
from preprocessing.Segmenter import Segmenter
from utils.Types import SegMode, SNR, Density


class TestSegmenter(unittest.TestCase):
    def test_segmenter_class(self):
        o_seg = Segmenter()
        self.assertEqual(o_seg.size, (512, 512, 10))
        self.assertEqual((o_seg.sigma, o_seg.kernel, o_seg.radius, o_seg.value), (1.0, 3, 1, 255))

        o_seg.update_size((512, 512, 20))
        o_seg.update_values(sigma=0.8)
        self.assertEqual(o_seg.size, (512, 512, 20))
        self.assertEqual((o_seg.sigma, o_seg.kernel, o_seg.radius, o_seg.value), (0.8, 3, 1, 255))

    def test_sphere_dts_without_slice(self):
        o_seg = Segmenter()
        o_seg.create_dataset(SegMode.sphere, SNR.TYPE_7, Density.LOW)

    def test_sphere_segmenter_with_slices(self):
        o_seg = Segmenter()
        o_seg.create_dataset(SegMode.sphere, SNR.TYPE_7, Density.LOW, save_img=True)

    def test_gauss_segmenter_with_sices(self):
        o_seg= Segmenter()
        o_seg.create_dataset(SegMode.gauss, SNR.TYPE_7, Density.LOW, save_img=True)


if __name__ == '__main__':
    unittest.main()
