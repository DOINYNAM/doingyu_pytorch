import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models

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
