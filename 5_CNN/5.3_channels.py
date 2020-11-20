import torch
import torch.nn as nn
from d2lzh_pytorch import corr2d

# 输入单批次多通道,输出单通道
def corr2d_multi_in(x, k):
    res = corr2d(x[0, :, :], k[0, :, :])
    for i in range(1, x.shape[0]):
        res += corr2d(x[i, :, :], k[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

y = corr2d_multi_in(X, K)
print(y)

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K]) # 将一个序列的tensor连接到一个新的维度

def pool2d(x, pool_size, mode = 'max'):
    x = x.float()
    p_h, p_w = pool_size
    h = x.shape[0] - p_h + 1
    w = x.shape[1] - p_w + 1
    y = torch.zeros(h,w)
    for i in range(h):
        for j in range(w):
            if mode == 'max':
                y[i][j] = torch.max(x[i: i + p_h, j: j + p_w])
            elif mode == 'avg':
                y[i][j] = x[i: i + p_h, j: j + p_w].mean()
    return y
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
y = pool2d(X, (2, 2))
print('max pooling result is \n', y)
y = pool2d(X, (2, 2), 'avg')
print('avg pooling result is \n', y)

x = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
pool = nn.MaxPool2d(kernel_size=3, stride=1)
y = pool(x)
print(y)

