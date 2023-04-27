import os
from typing import Tuple, List
import xml.etree.ElementTree as ET
import numpy as np
from cv2 import cv2
from Particle import Particle

PATH_GTH = "./Dataset/Challenge/ground_truth"
PATH_SLICES = "./Dataset/Challenge/VIRUS"


def create_video(dts_path: str):
    """Create Video from files (slices)

    :param dts_path:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("video.mp4", fourcc, fps=15, size=(512, 512))
    for frame_name in os.listdir(dts_path):
        frame = cv2.imread(os.path.join(dts_path, frame_name))
        out.write(frame.astype('uint8'))


def extract_particles(xml: str) -> List[Particle]:
    """Extract particles detected in xml file

    :param xml: file gth
    :return: list of detected particles
    """
    particles = []
    tree = ET.parse(xml)
    root = tree.getroot()
    for i, particle in enumerate(root[0]):
        for detection in particle:
            t, x, y, z = detection.attrib['t'], detection.attrib['x'], detection.attrib['y'], detection.attrib['z']
            particles.append(Particle(int(t), float(x), float(y), float(z)))
    return particles


def query_particles(particles: List[Particle],
                    t: int = None,
                    x: float = None,
                    y: float = None,
                    z: float = None) -> List[Particle]:
    """Get particles with criteria

    :param particles: list of particles
    :param t: desired t (None)
    :param x: desired x (None)
    :param y: desired y (None)
    :param z: desired z (None)
    :return: list of particles that satisfies criteria
    """
    particles_criteria = []
    if t is not None:
        particles_criteria = [particle for particle in particles if particle.t == t]
    if x is not None:
        particles_criteria = [particle for particle in particles if particle.x == x]
    if y is not None:
        particles_criteria = [particle for particle in particles if particle.y == y]
    if z is not None:
        particles_criteria = [particle for particle in particles if particle.z == z]
    return particles_criteria


def draw_particles(particles: List[Particle], frame: np.ndarray) -> np.ndarray:
    """Draw input particles on frame

    :param particles:
    :param frame:
    :return: frame with drawed particles
    """
    for particle in particles:
        x = round(particle.x)
        y = round(particle.y)
        cv2.circle(frame, (int(x), int(y)), 1, (60, 20, 220))
        cv2.putText(frame, text=particle.z, org=(x, y), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0, 255, 0))
    return frame



