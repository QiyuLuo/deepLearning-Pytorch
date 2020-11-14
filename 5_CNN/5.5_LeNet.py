import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from torch import nn, optim
import sys
sys.path.append('../3_basic')
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
batch_size = 256
lr = 0.001
num_epochs = 10

# 两个卷积(5*5)，池化(2*2),步长2，3个全连接层
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            nn.Sigmoid(),
        )
    def forward(self, img):
        feature = self.conv(img)
        # print(feature.shape)
        output = self.fc(feature.view(img.shape[0], -1)) # 将卷积得到的feature map铺平，每个样本一个向量得到size=(样本数, 特征数)
        return output

net = LeNet()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr)
if __name__ == '__main__':
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
