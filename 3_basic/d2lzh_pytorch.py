from IPython import display
from matplotlib import pyplot as plt
import torch
import random
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
import time
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        # 通道数看做特征数，长*宽看做样本数，将所有样本在每个通道上分别求平均值，得到最后的每个特征的概率。(在输出层即预测的种类概率)
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# 接收数组x, 核数组k,输出数组y. 单通道
def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros(x.shape[0] - h + 1, x.shape[1] - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = (x[i: i + h, j: j + w] * k).sum()
    return y

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize= (3.5,2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# 生成batch_size数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indexs = list(range(num_examples))
    random.shuffle(indexs) # 随机读取样本
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indexs[i:min(i + batch_size, num_examples)]) # 最后1批可能不够一个batchsize
        yield features.index_select(0, j), labels.index_select(0, j) # 选取feature,label第0维的j所表示的下标组，共batch_size个（除了最后一组可能不是），下次迭代接着执行。

# 矩阵乘，X-> batch_size * 2, w-> 2 * 1,
def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数

def squared_loss(y_hat, y):
    # print('y size = ', y.size())
    # print('y_hat size = ', y_hat.size())
    # return (y.view(y_hat.size()) - y_hat) ** 2 / 2 # 需要将真实标签的形状转换成预测张量的形状
    return (y_hat - y.view(y_hat.size())) ** 2 / 2 # y_hat_size = (batch_size, 1) h_size = (batch_size)  若不化成统一的形状， 会变成(batch_size, batch_size)的张量


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# 在一行里画出多张图像和对应标签的函数

def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size, resize=None):
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root='../../data', train=True, download=True,
                                        transform=transform)
    mnist_test = datasets.FashionMNIST(root='../../data', train=False, download=True,
                                       transform=transform)
    print('fashionMnist train len = ', len(mnist_train), mnist_train.data.shape)
    print('fashionMnist test  len = ', len(mnist_test))
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def accuracy(y_hat, y):
    return y_hat.argmax(dim=1).eq(y).float().sum().item() # 返回一个数值

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i, (feature, label) in enumerate(data_iter):
            if isinstance(net, torch.nn.Module):
                net.eval()
                feature = feature.to(device)
                label = label.to(device)
                output = net(feature)
                acc_sum += output.argmax(dim = 1).eq(label).sum().item()
            else: # 自定义的模型
                output = net(feature)
                acc_sum += accuracy(output, label)
            n += label.shape[0]
    return acc_sum / n

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):

        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        for feature, label in train_iter:
            output = net(feature)
            l = loss(output, label).sum() # 得到的是所有样本的损失
            l.backward() # loss后向传播，然后更新参数，最后将梯度清零
            # 更新参数
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
            # 梯度清零
            if optimizer is not None: # 如果是用pytorch
                optimizer.zero_grad()
            elif params is not None and params[0] is not None: # 自己写的
                for param in params:
                    param.grad.data.zero_()

            train_l_sum += l.item() # 计算损失和
            train_acc_sum += output.argmax(dim=1).eq(label.view(-1)).sum().item()
            if n == 0:
                print('label size = ', label.shape, ' output.argmax(dim=1) size = ', output.argmax(dim=1).shape)
            n += label.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item() # 得到训练的损失，均值
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


