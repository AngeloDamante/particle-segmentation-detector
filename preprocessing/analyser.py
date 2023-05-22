""" Utils Function to implement preprocessing phase """

import os
from typing import Tuple, List, Callable
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from preprocessing.Particle import Particle
from utils.Types import SNR, Density, SegMode
from utils.compute_path import get_gth_xml_path, get_slice_path, get_seg_data_path, get_seg_slice_path
from utils.definitions import DTS_CHALLENGE


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
    path_img = get_slice_path(snr, density, t, depth)
    path_gth = get_gth_xml_path(snr, density)

    # depth - eps =< z < depth + eps
    particles = extract_particles(path_gth)
    particles = query_particles(particles,
                                (lambda p: True if ((depth + eps > p.z >= depth - eps) and p.t == t) else False))

    # select img
    img = cv2.imread(path_img)
    if particles: img = draw_particles(particles, img.copy())
    return img, particles


def make_npy(snr: SNR, density: Density, t: int):
    depth = 10
    for t in range(t):
        img_list = []
        for z in range(depth):
            im = cv2.imread(get_slice_path(snr, density, t, z), 0)
            img_list.append(im)
        img_3d = np.stack(img_list, axis=2)
        path = os.path.join(DTS_CHALLENGE, 'VIRUS_npy', f'VIRUS_{snr}_{density}_npy')
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, f't_{str(t).zfill(3)}'), img_3d)

pass


def img_3d_comparator(mode: SegMode, snr: SNR, density: Density, t: int) -> bool:
    seg_path = get_seg_data_path(mode, snr, density, t)
    if not os.path.isfile(seg_path): return False
    img_3d = np.load(seg_path)

    num_cmp = 2
    plt.figure(figsize=(8.0, 5.0))
    for z in range(num_cmp):
        plt.subplot(2, num_cmp, z + 1)
        dts_slice = Image.open(get_slice_path(snr, density, t, z))
        plt.title(f'original {z}')
        plt.imshow(dts_slice, cmap="gray")
    for depth in range(num_cmp):
        plt.subplot(2, num_cmp, depth + num_cmp + 1)
        seg_slice = Image.open(get_seg_slice_path(mode, snr, density, t, depth))
        # seg_slice = Image.fromarray(img_3d[:, :, depth].astype(np.uint8))
        plt.title(f'segmap {depth}')
        plt.imshow(seg_slice, cmap="gray")
    plt.tight_layout()
    plt.savefig(f'{mode.value}_{snr.value}_{density.value}_{t}.tiff')
    return True
