""" Utils Function to implement preprocessing phase """

import os
import cv2
from typing import List, Callable
import numpy as np
import torch
import torchvision
import xml.etree.ElementTree as ET
from utils.Types import Particle


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


def query_particles(particles: List[Particle], criteria: Callable[[Particle], int]) -> List[Particle]:
    """Query on given input particles

    :param particles:
    :param criteria:
    :return: particles that respect a certain input criteria
    """
    particles_criteria = [particle for particle in particles if criteria(particle)]
    return particles_criteria


def draw_particles(img_3d: np.ndarray, particles: List[Particle]) -> np.ndarray:
    """Draw particles in 3d image to verify the correctness of particles

    :param img_3d: (H, W, D)
    :param particles:
    :return:
    """
    img = img_3d.copy()
    WHITE = 255
    for particle in particles:
        x = np.clip(round(particle.x), 0, img_3d.shape[0] - 1)
        y = np.clip(round(particle.y), 0, img_3d.shape[1] - 1)
        z = np.clip(round(particle.z), 0, img_3d.shape[2] - 1)
        _slice = img[:, :, z].astype(np.uint8)
        cv2.circle(_slice, center=(x, y), radius=5, color=WHITE)
        cv2.putText(_slice, text=str(particle.z), org=(x, y), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=WHITE)
        img[:, :, z] = _slice
    return img


def comparison(x: np.ndarray, y: np.ndarray, save_dir: str):
    """Comparison viewer between x and y

    :param x:
    :param y:
    :param save_dir:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    for z in range(x.shape[2]):
        cmp_img = np.hstack((x[:, :, z], y[:, :, z]))
        cv2.imwrite(os.path.join(save_dir, f'{str(z).zfill(3)}.png'), cmp_img)
