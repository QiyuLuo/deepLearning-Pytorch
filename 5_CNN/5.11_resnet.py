import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append('../3_basic')
import d2lzh_pytorch as d2l

batch_size = 256
lr = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 得到的形状不变，通道为out_channels
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv: # 用来改变通道或者长宽
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)

# 这里每个模块里有4个卷积层（不计算1×1卷积层），加上最开始的卷积层和最后的全连接层，共计18层。这个模型通常也被称为ResNet-18。
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 为ResNet加入残差块，每个模块使用两个残差快,由于第一个模块之前pooling步幅为2所以第一个模块不需要高宽减半，
# 其余模块channel加倍，高宽减半。
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))


x = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    x = layer(x)
    print(name, ' output shape:\t', x.shape)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
optimizer = torch.optim.Adam(net.parameters(), lr)
if __name__ == '__main__':
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)