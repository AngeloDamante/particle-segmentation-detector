""" Utils Function to implement preprocessing phase """

import os
from typing import Tuple, List, Callable
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from preprocessing.Particle import Particle
from utils.Types import  SNR, Density
from utils.compute_path import get_gth_path, get_img_path


def create_video(dts_path: str):
    """Create Video from files (slices)

    :param dts_path:
    :return:
    """
    if not os.path.exists(dts_path): return
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
    if not os.path.isfile(xml): return []
    particles = []
    tree = ET.parse(xml)
    root = tree.getroot()
    for i, particle in enumerate(root[0]):
        for detection in particle:
            t, x, y, z = detection.attrib['t'], detection.attrib['x'], detection.attrib['y'], detection.attrib['z']
            particles.append(Particle(int(t), float(x), float(y), float(z)))
    return particles


def query_particles(particles: List[Particle], criteria: Callable[[Particle], int]):
    """Query on given input particles

    :param particles:
    :param criteria:
    :return: particles that respect a certain input criteria
    """
    particles_criteria = [particle for particle in particles if criteria(particle)]
    return particles_criteria


def draw_particles(particles: List[Particle], frame: np.ndarray) -> np.ndarray:
    """Draw input particles on frame

    :param particles:
    :param frame:
    :return: frame with drawed particles
    """
    if frame is None: return 255 * np.ones((512, 512, 3), np.uint8)
    if not particles: return frame
    for particle in particles:
        x = round(particle.x)
        y = round(particle.y)
        cv2.circle(frame, (int(x), int(y)), 1, (60, 20, 220))
        cv2.putText(frame, text=str(particle.z), org=(x, y), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0, 255, 0))
    return frame


def draw_particles_slice(snr: SNR, t: int, depth: int, density: Density, eps: float = 2.0) -> Tuple[
    np.ndarray, List[Particle]]:
    """Draw particles with criteria to select slice

    :param snr:
    :param t:
    :param depth:
    :param density:
    :param eps:
    :return:
    """
    path_img = get_img_path(snr, density, t, depth)
    path_gth = get_gth_path(snr, density)

    # depth - eps =< z < depth + eps
    particles = extract_particles(path_gth)
    particles = query_particles(particles, (lambda p: True if ((depth + eps > p.z >= depth - eps) and p.t == t) else False))

    # select img
    img = cv2.imread(path_img)
    if particles: img = draw_particles(particles, img.copy())
    return img, particles
