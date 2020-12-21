import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()

img = Image.open('../img/cat1.jpg')

d2l.apply(img, torchvision.transforms.RandomHorizontalFlip())

d2l.apply(img, torchvision.transforms.ColorJitter())