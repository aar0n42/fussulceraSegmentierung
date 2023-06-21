import torch
import os
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch

allimages = os.listdir("images/")

model = torch.load('model.pth',map_location=torch.device('cpu'))

transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()


for i in allimages:
    img_loc = 'images/' + i
    image = Image.open(img_loc)
    image = transform(image)
    image = image.unsqueeze(0)
    #image = image.to('cuda')
    prediction = model(image)

    prediction = prediction.squeeze()
    prediction = (prediction>0.5).float()
    image_prediction = transformToPIL(prediction)
    image_prediction.save("/predictions/" + i)
    #prediction = model(image.to('cuda'))
