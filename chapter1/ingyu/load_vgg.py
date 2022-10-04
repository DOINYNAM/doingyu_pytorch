from matplotlib import transforms
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

# print("PyTorch Version ", torch.__version__)
print("torchvision version ", torchvision.__version__)
print(np)
print(json)
print(Image)
print(plt)
print(torch)

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval()

print(net)


class BaseTransform():
    def __init__(self, resize, mean, std) -> None:
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img) -> np.array:
        return self.base_transform(img)


print(BaseTransform)
