import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

# 生成数据
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.mm(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float) # 加噪音
train_features, test_features = features[:n_train, :], features[n_train:, :] # 取训练和测试样本
train_labels, test_labels = labels[:n_train, :], labels[n_train:, :]

def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚项
def l2_penalty(w):
    return (w ** 2).sum() / 2

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = data.TensorDataset(train_features, train_labels)
train_iter = data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for feature, label in train_iter:
            output = net(feature, w, b)
            l = loss(output, label) + lambd * l2_penalty(w)
            l = l.sum()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())

def fit_and_plot_pytorch(lambd):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=lambd) # 只对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr) # 不对偏差参数衰减
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for feature, label in train_iter:
            output = net(feature)
            l = loss(output, label)
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


if __name__ == '__main__':
    # fit_and_plot(3)
    fit_and_plot_pytorch(3)