import unittest
from preprocessing.analyser import extract_particles, query_particles
from utils.Types import SNR, Density
from utils.compute_path import get_gth_xml_path


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


if __name__ == '__main__':
    unittest.main()
