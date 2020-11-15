import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append('../3_basic')
import d2lzh_pytorch as d2l

batch_size = 64
lr = 0.002 # 学习率更大
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# nin块由一个卷积层和充当全连接的两个1*1卷积层构成。
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk

class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        # 通道数看做特征数，长*宽看做样本数，将所有样本在每个通道上分别求平均值，得到最后的每个特征的概率。(在输出层即预测的种类概率)
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
"""
NiN重复使用由卷积层和代替全连接层的1×11×1卷积层构成的NiN块来构建深层网络。
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
NiN的以上设计思想影响了后面一系列卷积神经网络的设计。
"""
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    d2l.FlattenLayer()
)
"""
x = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    x = blk(x)
    print(name, 'output shape: ', x.shape)
"""

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
optimizer = optim.Adam(net.parameters(), lr)

if __name__ == '__main__':
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)