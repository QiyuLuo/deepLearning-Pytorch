import torch
import torch.nn as nn

# 接收数组x, 核数组k,输出数组y
def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros(x.shape[0] - h + 1, x.shape[1] - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = (x[i: i + h, j: j + w] * k).sum()
    return y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
y = corr2d(X, K)
print(y)

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

x = torch.ones(6, 8)
x[:, 2:6] = 0

# 构造一个kernel,使得当相邻两个元素相同时，结果为0，否则非0
k = torch.tensor([[1, -1]])

y = corr2d(x, k)
print(y)
lr = 0.6
epochs = 20
net = Conv2D(kernel_size=(1, 2))
# loss = nn.MSELoss(reduction='sum')
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr= lr)
def train():
    for epoch in range(epochs):
        output = net(x)
        l = loss(output, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 5 == 0:
            print('train1 Step %d, loss %.3f' % (epoch + 1, l.item()))
def train2():
    for epoch in range(epochs):
        Y_hat = net(x)
        l = ((Y_hat - y) ** 2).sum()
        l.backward()

        # 梯度下降
        net.weight.data -= lr * net.weight.grad
        net.bias.data -= lr * net.bias.grad

        # 梯度清0
        net.weight.grad.fill_(0)
        net.bias.grad.fill_(0)
        if (epoch + 1) % 5 == 0:
            print('train2 Step %d, loss %.3f' % (epoch + 1, l.item()))

if __name__ == '__main__':
    train()
    # train2()
    print(net.weight.data)
    print(net.bias.data)

"""
当使用MSE loss并设置reduction=mean时，学习率通常要比设置成sum大一些，否则梯度下降的比较慢。   lr = 0.6
当使用MSE loss并设置reduction=sum时，学习率通常要设置小一点，否则可能会导致梯度爆炸。   lr = 0.01

"""