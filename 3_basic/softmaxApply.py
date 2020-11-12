import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

batch_size = 256
num_inputs = 784 # 28*28
num_outputs = 10
num_epochs = 5
lr = 0.6
# 读取数据

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

"""

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.ones((2, 1)) * 2

x1 = torch.tensor([[1], [2], [3]])
y1 = torch.tensor([1,2,3])
x.view()
print(x1 + y1)

# keepdim = True会保持原有的维度个数，即原来是二维张量(2,3)，计算完后仍然是二维张量，
# keepdim = Fasle 会变成1维张量
print(x.sum(dim=0, keepdim=False))
print(x.sum(dim=0, keepdim=True))
print(x.sum(dim=1, keepdim=True))
x = torch.tensor([2])

print(x.sum(dim=1, keepdim=False))
print(x / y)
"""
# 定义softmax运算
def softmax(x):
    x_exp = x.exp() # 对输出的各个元素做指数运算
    partition = x_exp.sum(dim=1, keepdim=True) # 求每一行的总和
    return x_exp / partition # 得到每个元素在输出类别上对应的概率

# 定义模型
def net(x):
    return softmax(torch.mm(x.view((-1, num_inputs)), w) + b)
# 定义损失函数，求的是一个batch_size的损失
def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1))) # gather聚合函数，将y_hat指定轴(行)对应的下标位置的值聚合起来

def accuracy(y_hat, y):
    return y_hat.argmax(dim=1).eq(y).float().mean().item() # 返回一个数值



if __name__ == '__main__':

    d2l.train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)