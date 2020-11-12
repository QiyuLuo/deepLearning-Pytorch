from IPython import display
from matplotlib import pyplot as plt
import torch
import random
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
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

def load_data_fashion_mnist(batch_size):
    mnist_train = datasets.FashionMNIST(root='../../data/FashionMNIST', train=True, download=True,
                                        transform=transforms.ToTensor())
    mnist_test = datasets.FashionMNIST(root='../../data/FashionMNIST', train=False, download=True,
                                       transform=transforms.ToTensor())
    print('fashionMnist train len = ', len(mnist_train))
    print('fashionMnist test  len = ', len(mnist_test))
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def accuracy(y_hat, y):
    return y_hat.argmax(dim=1).eq(y).float().mean().item() # 返回一个数值

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for i, (feature, label) in enumerate(data_iter):
        output = net(feature)
        acc_sum += accuracy(output, label)
        n += 1
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
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))



