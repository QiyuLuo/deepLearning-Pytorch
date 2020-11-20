import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append('../3_basic')
import d2lzh_pytorch as d2l

batch_size = 128
lr = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前是训练还是测试
    if not is_training:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            # 使用全连接层,得到每一列的均值和方差，即每个特征的均值和方差
            mean = x.mean(dim = 0)
            var = ((x - mean) ** 2).mean(dim = 0)
        else:
            # 二维卷积层,计算通道维(dim=1)上的均值和方差，需要保持x的形状做广播运算
            mean = x.mean(dim = 0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) # shape = (1*C*1*1)
            var = ((x - mean) ** 2).mean(dim = 0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) # shape = (1*C*1*1)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    y = gamma * x_hat + beta # 拉伸和偏移即仿射变换
    return y, moving_mean, moving_var
"""
在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络的中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。
对全连接层和卷积层做批量归一化的方法稍有不同。
批量归一化层和丢弃层一样，在训练模式和预测模式的计算结果是不一样的。
PyTorch提供了BatchNorm类方便使用。
"""
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移系数，分别初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化为0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, x):
        # 如果x不在内存上，将moving_mean和moving_var放在显存上，因为这两个参数不是Parameter类型，模型不会将其自动放入gpu上
        self.moving_mean = self.moving_mean.to(x.device)
        self.moving_var = self.moving_var.to(x.device)

        # 保存更新过的moving_mean,和moving_var, Module实例的training属性默认为True,调用eval后设为false
        y, self.moving_mean, self.moving_var = batch_norm(
            self.training, x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return y
x = torch.rand(3, 4)
print(x)
print(x.mean(dim=0, keepdim=True))
print(x.mean(dim=1, keepdim=True))

# 使用自己实现批量归一化层的LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),  # kernel_size, stride

    nn.Conv2d(6, 16, 5),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    d2l.FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),

    nn.Linear(84, 10)
)
# pytorch 实现的BN
net2 = nn.Sequential(
    nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),  # kernel_size, stride

    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),

    d2l.FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    nn.BatchNorm1d(120), # 全连接要用1d
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),

    nn.Linear(84, 10)
)
# x = torch.rand(2, 1, 28, 28)
# for name, blk in net.named_children():
#     x = blk(x)
#     print('name: ', name, 'output shape: ', x.shape)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr)
"""
y = torch.rand(1, 10)
bn = nn.BatchNorm1d(10)
output = bn(y)
print('y=', y)
# 样本数必须大于1
"""
if __name__ == '__main__':
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    print(net[1].gamma.view((-1,)), '\n', net[1].beta.view((-1,)))