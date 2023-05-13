import unittest
import os
import cv2
from utils.definitions import DTS_VIRUS, DTS_GTH
from preprocessing.analyser import extract_particles, query_particles, draw_particles, draw_particles_slice
from utils.compute_path import get_gth_path, get_img_path
from preprocessing.Particle import Particle
from utils.Types import SNR, Density
from typing import Tuple, List


class PreprocessingUT(unittest.TestCase):
    my_virus : str = "VIRUS_snr_7_density_low"
    my_particles: List[Particle] = []
    my_path_xml: str = os.path.join(DTS_GTH, f'{my_virus}.xml')
    my_path_img: str = os.path.join(DTS_VIRUS, f'{my_virus}', f'{my_virus}_t000_z00.tif')

    def test_extract_particles(self):
        self.my_particles = extract_particles(self.my_path_xml)
        self.assertEqual(len(self.my_particles), 9858)

    def test_query_particles(self):
        # select only t = 0
        self.my_particles = extract_particles(self.my_path_xml)
        my_particles_t0 = query_particles(self.my_particles, (lambda p: True if p.t == 0 else False))
        self.assertEqual(len(my_particles_t0), 74)

        # select only z in [0,1)
        my_particles_z0 = query_particles(self.my_particles, (lambda p: True if 1 > p.z >= 0 else False))
        self.assertGreaterEqual(min([particle.z for particle in my_particles_z0]), 0)
        self.assertLess(max([particle.z for particle in my_particles_z0]), 1)

    def test_draw_particles(self):
        self.my_particles = extract_particles(self.my_path_xml)
        img = cv2.imread(self.my_path_img)
        frame = draw_particles(self.my_particles, img)
        cv2.imwrite("img_test.png", frame)

    def test_select_slice(self):
        # identifying tuple
        idx = (SNR.TYPE_7, Density.LOW, 1, 2)
        path_file_img = get_img_path(*idx)
        path_file_gth = get_gth_path(idx[0], idx[1])

        self.assertTrue(os.path.isfile(path_file_img))
        self.assertTrue(os.path.isfile(path_file_gth))

    def test_draw_particles_slice(self):
        snr, t, depth, density = SNR.TYPE_7, 0, 8, Density.LOW
        img, self.my_particles = draw_particles_slice(snr, t, depth, density)
        self.assertTrue(self.my_particles)
        cv2.imwrite("slice_test.png", img)

if __name__ == '__main__':
    unittest.main()
