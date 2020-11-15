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
            # 使用全连接层,得到每一列的均值，即每个特征的均值
            mean = x.mean(dim = 0)
            var = ()
x = torch.rand(3, 4)
print(x)
print(x.mean(dim=0))
print(x.mean(dim=1))
