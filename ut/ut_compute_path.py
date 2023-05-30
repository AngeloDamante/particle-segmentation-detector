import unittest
import os
from utils.compute_path import compute_name, get_data_path, get_slice_path, get_gth_xml_path
from utils.Types import SNR, Density
from utils.definitions import (
    DEFAULT_DATA_PATH,
    DEFAULT_SLICES_PATH,
    DEFAULT_GTH_PATH,
    DTS_RAW_PATH
)


class ComputePathTest(unittest.TestCase):
    def test_compute_name(self):
        self.assertEqual(compute_name(SNR.TYPE_7, Density.LOW), "snr_7_density_low")
        self.assertEqual(compute_name(SNR.TYPE_7, Density.LOW, t=1), "snr_7_density_low_t001")
        self.assertEqual(compute_name(SNR.TYPE_7, Density.LOW, t=1, z=4), "snr_7_density_low_t001_z04")
        self.assertEqual(compute_name(SNR.TYPE_7, Density.LOW, z=3), "snr_7_density_low_z03")

    def test_slice_path(self):
        path = get_slice_path(SNR.TYPE_7, Density.LOW, t=0, z=0)
        self.assertEqual(path, os.path.join(DEFAULT_SLICES_PATH, "snr_7_density_low", "snr_7_density_low_t000_z00.tif"))
        path = get_slice_path(SNR.TYPE_7, Density.LOW, t=1, z=5, root=DEFAULT_SLICES_PATH)
        self.assertEqual(path, os.path.join(DEFAULT_SLICES_PATH, "snr_7_density_low", "snr_7_density_low_t001_z05.tif"))

    def test_data_path(self):
        path = get_data_path(SNR.TYPE_7, Density.LOW, t=1)
        self.assertEqual(path, os.path.join(DEFAULT_DATA_PATH, "snr_7_density_low", "snr_7_density_low_t001.npy"))
        path = get_data_path(SNR.TYPE_7, Density.LOW, t=1, is_npz=True, root=DTS_RAW_PATH)
        self.assertEqual(path, os.path.join(DTS_RAW_PATH, "snr_7_density_low", "snr_7_density_low_t001.npz"))

    def test_gth_path(self):
        path = get_gth_xml_path(SNR.TYPE_7, Density.LOW)
        self.assertEqual(path, os.path.join(DEFAULT_GTH_PATH, "snr_7_density_low.xml"))


if __name__ == '__main__':
    unittest.main()
