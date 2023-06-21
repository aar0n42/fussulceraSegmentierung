import numpy as np
import cv2 as cv
from PIL import Image
import cv2
import numpy
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)
mini = 1000
maxx = 0
summe = 0
count = 0

for label in os.listdir("data/validation/labels/"):
    tsr = torchvision.io.read_image('data/validation/labels/' + label)
    tsr = asdf(tsr)
    count += 1
    a = tsr.unique(return_counts=True)
    if a[1][0].item() == 262144: #wenn alle pixel 0 sind
        continue
    area = a[1][1].item()
    summe += area
    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area

print(loc)
print(mini)
print(f)
print(maxx)

print(count)

mini = 1000
maxx = 0
"""

for label in os.listdir("data/train/labels/"):
    tsr = torchvision.io.read_image('data/train/labels/' + label)
    tsr = asdf(tsr)
    a = tsr.unique(return_counts=True)
    count += 1
    if a[1][0].item() == 262144: #wenn alle pixel 0 sind
        continue
    area = a[1][1].item()
    summe += area
    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area
print(count)

print()
print(loc)
print(mini)
print(f)
print(maxx)
print()
"""
#https://stackoverflow.com/questions/59525065/detecting-object-location-in-image-with-python-opencv object location