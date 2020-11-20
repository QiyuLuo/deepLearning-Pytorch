import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from torch import nn, optim
import sys
sys.path.append('../3_basic')
import d2lzh_pytorch as d2l

batch_size = 64
lr = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) # 卷积尺寸保持不变
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 池化后高宽减半
    return nn.Sequential(*blk)

# 定义卷积层模块，指定了每个VGG块的卷积层个数，输入输出通道个数。
# 前两个单卷积层，后3个双卷积层，每次卷积后通道翻倍，从64到512，图像的高和宽减半, 224 / 2^5 = 7。3个全连接层。所以共11层
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))

fc_features = 512*7*7
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module('fc', nn.Sequential(
        d2l.FlattenLayer(), # 得到的特征图，需要将特征展开成一维向量
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
optimizer = torch.optim.Adam(net.parameters(), lr)
if __name__ == '__main__':
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
