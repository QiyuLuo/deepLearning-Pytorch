import torch
import torchvision.datasets as datasets
import torch.nn as nn
import d2lzh_pytorch as d2l
import torch.optim as optim
batch_size = 256
lr = 0.1
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 5
# 定义模型
class mlp(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(mlp, self).__init__()
        self.FlattenLayer = d2l.FlattenLayer()
        self.Linear1 = nn.Linear(num_inputs, num_hiddens)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = self.FlattenLayer(x)
        x = self.Linear1(x)
        x = self.ReLU(x)
        y = self.Linear2(x)
        return y

net = mlp(num_inputs, num_hiddens, num_outputs)

# 初始化参数

for params in net.parameters():
    nn.init.normal_(params, mean=0, std=0.01)

# 加载数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)


if __name__ == '__main__':
    d2l.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)