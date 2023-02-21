import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob as glob
from torch import topk
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import sys, os
import matplotlib.image as mpimg

batch_size = 4
device = ('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(500544, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
PATH = './hand_net_4.pth'


model.load_state_dict(torch.load(PATH))
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((960, 540)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.ImageFolder(root='check', transform=transform)

classes = ('power', 'precision')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

target_layers = [model.conv2]

for path, subdirs, files in os.walk('frames'):
    for name in files:
        image_path = os.path.join(path, name)
        
        image = cv2.imread(image_path)
        orig_image = image.copy()
        height, width, _ = orig_image.shape
        # apply the image transforms
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)

        from PIL import Image
        img = np.array(Image.open(image_path))

        # plt.imshow(img, vmin=0, vmax=1)
        # plt.show()

        img = transform(img)
        cam = GradCAM(model=model, target_layers=target_layers)

        grayscale_cam = cam(input_tensor=input_tensor)

        img = np.array(img).reshape(960,540,3)
        # plt.imshow(img, vmin=0, vmax=1)
        # plt.show()

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        # print(img.shape)
        # img = rgb2gray(img)


        # img = mpimg.imread('check/precision/precision0030.png')     
        # gray = rgb2gray(img)    

        img = np.array(Image.open(image_path))

        transforms2 = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize((960, 540))
        ])
        img = np.array(transforms2(img))/255



        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # print('**********************************')
        # print((gray.shape))
        visualization = show_cam_on_image(np.array(img), grayscale_cam, use_rgb=True)

        resized_img = cv2.resize(visualization, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(resized_img, vmin=0, vmax=1)
        # plt.show()
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

        if 'precision' in image_path:
            cv2.imwrite(f'cam_results/precision/{name}', resized_img)

        else:
            cv2.imwrite(f'cam_results/power/{name}', resized_img)
