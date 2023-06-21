from torchsummary import summary
import enum
import numpy as np
import cv2 as cv
from PIL import Image
import cv2
import numpy
import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision


encoder = 'resnet34'
encoder_weights = 'imagenet'


model = smp.Unet(
    encoder_name=encoder, 
    encoder_weights=encoder_weights, 
    classes=1,
    in_channels=3,
    encoder_depth = 5,
    decoder_channels = [256, 128, 64, 32, 16],
)  

model.to('cuda')
print(summary(model, (3,512,512)))