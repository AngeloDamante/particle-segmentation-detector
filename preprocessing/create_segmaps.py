import os
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
from preprocessing.analyser import extract_particles, query_particles, get_img_path, get_gth_path
from preprocessing.loader import get_slice
from preprocessing.Particle import Particle, SNR, Density
from utils.definitions import DTS_SEG_1

SIZE_VOL = (512, 512, 10)


# technique 1
def make_segmaps_with_spheres(snr: SNR, density: Density, radius: int = 1, value: int = 1) -> bool:
    """Segmentation map maker

    :param snr:
    :param density:
    :param radius:
    :param value:
    :return:
    """
    # take particle
    path_gth = get_gth_path(snr, density)
    particles = extract_particles(path_gth)
    if len(particles) == 0: return False

    # make 3d images for each t
    for time in tqdm(range(100)):
        # get particles at time t
        particles_t = query_particles(particles, (lambda pa, time=time: True if pa.t == time else False))

        # make blank 3d image
        img_3d = np.zeros(shape=SIZE_VOL)
        x, y, z = np.meshgrid(np.arange(img_3d.shape[0]), np.arange(img_3d.shape[1]), np.arange(img_3d.shape[2]))

        # draw spheres centered in known particles
        for p in particles_t:
            center = (round(p.x), round(p.y), np.clip(round(p.z), 0, 9))
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
            mask = (distance <= radius)
            img_3d[mask] = value

        # save slices
        dir_name = f'segVirus_{snr.value}_{str(density.value)}'
        target_path = os.path.join(f'{DTS_SEG_1}', dir_name)
        if not (os.path.isdir(target_path)): os.mkdir(target_path)
        for depth in range(10):
            img_name = f't_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tiff'

            # cv2
            img = img_3d[:, :, depth]
            cv2.imwrite(os.path.join(target_path, img_name), img)

            # pil
            # img = Image.fromarray(img_3d[:, :, depth])
            # img.save(os.path.join(target_path, img_name))
    return True


# technique 2
def make_segmaps_gussianblur(snr: SNR, density: Density, value: int = 255, sigma: tuple = (0.1, 2.0),
                             kernel_size: tuple = (3, 3)) -> bool:
    # take particle
    path_gth = get_gth_path(snr, density)
    particles = extract_particles(path_gth)
    if len(particles) == 0: return False

    # make 3d images for each t
    for time in tqdm(range(100)):
        # get particles at time t
        particles_t = query_particles(particles, (lambda pa, time=time: True if pa.t == time else False))

        # make blank 3d image
        img_3d = np.zeros(shape=SIZE_VOL)

        for p in particles_t:
            center = (round(p.x), round(p.y), np.clip(round(p.z), 0, 9))
            img_3d[center] = value

        # GaussianBlur
        convert_tensor = transforms.ToTensor()
        img_3d = convert_tensor(img_3d)
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        img = transform(img_3d)

        # save slices
        dir_name = f'segVirusGauss_{snr.value}_{str(density.value)}'
        target_path = os.path.join(f'{DTS_SEG_1}', dir_name)
        if not (os.path.isdir(target_path)): os.mkdir(target_path)
        for depth in range(10):
            img_name = f't_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tiff'

            # cv2
            imgblur = transforms.ToPILImage()(img[depth, :, :])
            imgblur.save(os.path.join(target_path, img_name))
    return True


# plt.figure()
# for i in range(4):
#     plt.subplot(2,4,i+1)
#     plt.title(f'image {i}')
#     plt.imshow(Image.fromarray(img_3d[:, :, i]), cmap="gray")
# for j in range(4):
#     plt.subplot(2,4,j+4+1)
#     plt.title(f'seg {j}')
#     plt.imshow(Image.fromarray(img_3d_t0[:, :, j]), cmap="gray")
# plt.savefig("comparison.png")


if __name__ == '__main__':
    make_segmaps_gussianblur(SNR.TYPE_7, Density.LOW)
#     make_segmaps_with_spheres(SNR.TYPE_7, Density.LOW, value=255)
