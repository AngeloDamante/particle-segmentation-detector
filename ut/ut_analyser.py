import unittest
import cv2
import os
import numpy as np
from preprocessing.analyser import extract_particles, query_particles, draw_particles
from utils.Types import SNR, Density
from utils.compute_path import get_gth_xml_path, get_data_path
from utils.definitions import DTS_RAW_PATH, DTS_ANALYZE_PATH


class PreprocessingUT(unittest.TestCase):

    def test_extract_particles(self):
        my_path_xml = get_gth_xml_path(SNR.TYPE_7, Density.LOW)
        my_particles = extract_particles(my_path_xml)
        self.assertEqual(len(my_particles), 9858)

    def test_query_particles(self):
        # select only t = 0
        my_path_xml = get_gth_xml_path(SNR.TYPE_7, Density.LOW)
        my_particles = extract_particles(my_path_xml)
        my_particles_t0 = query_particles(my_particles, (lambda p: True if p.t == 0 else False))
        self.assertEqual(len(my_particles_t0), 74)

        # select only z in [0,1)
        my_particles_z0 = query_particles(my_particles, (lambda p: True if 1 > p.z >= 0 else False))
        self.assertGreaterEqual(min([particle.z for particle in my_particles_z0]), 0)
        self.assertLess(max([particle.z for particle in my_particles_z0]), 1)

    def test_draw_particles(self):
        data = np.load(get_data_path(SNR.TYPE_7, Density.LOW, t=1, is_npz=True, root=DTS_RAW_PATH), allow_pickle=True)
        img_3d = data['img']
        gth = list(data['gth'])
        img_3d_with_particles = draw_particles(img_3d, gth)
        for i in range(img_3d_with_particles.shape[2]):
            cv2.imwrite(os.path.join(DTS_ANALYZE_PATH, f"slice_{i}.png"), img_3d_with_particles[:, :, i])
        self.assertTrue(gth)
        self.assertEqual(img_3d.shape, img_3d_with_particles.shape)


if __name__ == '__main__':
    unittest.main()
