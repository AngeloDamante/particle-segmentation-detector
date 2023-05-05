import os
from PIL import Image
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from preprocessing.analyser import PATH_GTH, PATH_IMG
from preprocessing.analyser import extract_particles, query_particles, get_img_path, get_gth_path
from preprocessing.loader import get_slice
from preprocessing.Particle import Particle, SNR, Density
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH_DTS_1 = "../Dataset/seg_technique_1"
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
        directory = f'{PATH_DTS_1}/segVirus_{snr.value}_{str(density.value)}'
        if not (os.path.isdir(directory)): os.mkdir(directory)
        for depth in range(10):
            img = Image.fromarray(img_3d[:, :, depth])
            img.save(f'{directory}/t_{str(time).zfill(3)}_z_{str(depth).zfill(2)}.tif')
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
    make_segmaps_with_spheres(SNR.TYPE_7, Density.LOW, value=255)
