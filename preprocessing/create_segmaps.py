import random
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from cv2 import cv2
from preprocessing.analyser import PATH_GTH, PATH_IMG
from preprocessing.analyser import extract_particles, query_particles
from preprocessing.loader import get_slice
from preprocessing.Particle import Particle, SNR, Density
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# blank image
size = (512, 512, 10)  # aumentare il numero di slice
seg_map = torch.zeros(size=size, dtype=torch.uint8)

# t0
particles = extract_particles(f'{PATH_GTH}/VIRUS_snr_7_density_low.xml')
particles_t0 = query_particles(particles, (lambda pa: True if pa.t == 0 else False))

# put spheres
for p in particles_t0:
    seg_map[round(p.x), round(p.y), np.clip(round(p.z), 0, 9)] = 255

# p_rand = random.choice(particles_t0)
# print(seg_map[round(p_rand.x), round(p_rand.y), round(p_rand.z)].item())

# take slices at t=0
slices_t0 = []
for depth in range(10):
    slices_t0.append(np.asarray(get_slice(SNR.TYPE_7, Density.LOW, t=0, depth=depth)))
img_3d = np.stack(slices_t0, axis=2)
print(img_3d.shape)

# take particles at t=0
particles = extract_particles(f'{PATH_GTH}/VIRUS_snr_7_density_low.xml')
particles_t0 = query_particles(particles, (lambda pa: True if pa.t == 0 else False))

# analyse
img_3d_t0 = img_3d.copy()
for p in particles_t0:
    x, y, z = round(p.x), round(p.y), np.clip(round(p.z), 0, 9)
    img_3d_t0[x, y, z] = 255
    img_3d_t0[:,:,z] = cv2.circle(img_3d_t0[:,:,z].astype(np.uint8), center=(x,y), radius=2, color=(255,255,255), thickness=cv2.FILLED)

# comparison
plt.figure()
for i in range(4):
    plt.subplot(2,4,i+1)
    plt.title(f'image {i}')
    plt.imshow(Image.fromarray(img_3d[:, :, i]), cmap="gray")
for j in range(4):
    plt.subplot(2,4,j+4+1)
    plt.title(f'seg {j}')
    plt.imshow(Image.fromarray(img_3d_t0[:, :, j]), cmap="gray")
plt.savefig("comparison.png")
# FIX!! non li carica in ordine, forse