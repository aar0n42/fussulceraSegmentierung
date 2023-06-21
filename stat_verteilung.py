import enum
import numpy as np
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
import numpy as np
import torchvision

plt.figure(figsize=(8, 6),dpi=600)
transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)
mini = 1000
maxx = 0
summe = 0
count = 0
alle = []
for label in os.listdir("data/train/labels/"):
    tsr = torchvision.io.read_image('data/train/labels/' + label)
    tsr = asdf(tsr)
    count += 1
    a = tsr.unique(return_counts=True)
    if a[1][0].item() == 262144: #wenn alle pixel 0 sind
        alle.append(0)
        continue

    area = a[1][1].item()
    summe += area
    alle.append(area)

    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area


alle.sort()

for i, num in enumerate(alle):
    alle[i] = round(num/262144,3)


print(alle)

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
j = 0
k = 0
l = 0
m = 0
n = 0
o = 0
y = 0
p = 0
for i in alle:
    if i < 0.005:
        a += 1
    elif  i < 0.01:
        b += 1
    elif i < 0.015:
        c += 1
    elif i < 0.02:
        d += 1
    elif i < 0.025:
        e += 1
    elif i < 0.03:
        f += 1
    elif i < 0.035:
        g += 1
    elif i < 0.04:
        h += 1
    elif i < 0.045:
        j += 1
    elif i < 0.05:
        k += 1
    elif i < 0.055:
        l += 1
    elif i < 0.06:
        m += 1
    else:
        y +=1

"""
alle2 = []
for label in os.listdir("data/validation/labels/"):
    tsr = torchvision.io.read_image('data/validation/labels/' + label)
    tsr = asdf(tsr)
    count += 1
    ass = tsr.unique(return_counts=True)
    if ass[1][0].item() == 262144: #wenn alle pixel 0 sind
        alle2.append(0)
        continue

    area = ass[1][1].item()
    summe += area
    alle2.append(area)

    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area


alle2.sort()

print(alle2)
for i, num in enumerate(alle2):
    alle2[i] = round(num/262144,3)

for i in alle2:
    if i < 0.005:
        a += 1
    elif  i < 0.01:
        b += 1
    elif i < 0.015:
        c += 1
    elif i < 0.02:
        d += 1
    elif i < 0.025:
        e += 1
    elif i < 0.03:
        f += 1
    elif i < 0.035:
        g += 1
    elif i < 0.04:
        h += 1
    elif i < 0.045:
        j += 1
    elif i < 0.05:
        k += 1
    elif i < 0.055:
        l += 1
    elif i < 0.06:
        m += 1
    else:
        y +=1
"""
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(j)
print(k)
print(l)
print(m)
print(n)
print(o)
print(p)
print(y)
print(a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+y)

print(len(alle))


height = [a,b,c,d,e,f,g,h,j,k,l,m,y]
bars = ('<0,5%', '0,5≥ <1%', '1≥ <1,5%', '1,5≥ <2%', '2≥ <2,5%', '2,5≥ <3%', '3≥ <3,5%', '3,5≥ <4%', '4≥ <4,5%', '4,5≥ <5%', '5≥ ≤5,5%', '5,5≥ <6', '≥6%')
y_pos = np.arange(len(bars))


plt.bar(y_pos, height)

plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.show()


"""

#print(loc)
#print(mini)
#print(f)
#print(maxx)

print(count)

mini = 1000
maxx = 0
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
