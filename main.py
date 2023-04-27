import torch
from cv2 import cv2

print(cv2.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
