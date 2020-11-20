import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as Data
from generateDataset import *
import torch.optim as optim
import torch.utils.data
import sys
import time
import d2lzh_pytorch as d2l
# 定义超参数

batch_size = 50
num_epochs = 3
log_interval = 10
lr = 0.3



torch.set_default_tensor_type(torch.FloatTensor)

all_features = pd.read_csv('E:/pythonProject/data/Android_Data.csv',header=None) # (1500, 887)


# 原始数据的第一个特征是id,不能作为特征来推断测试集所以总特征只有79个，最后一个是价格
print(all_features.shape)
# print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print(type(all_features.values))

# print(len(all_features.dtypes[all_features.dtypes == 'object'].index))

# print(all_features.iloc[8:13, [1, 2, 3, 4, 5, 6, -3, -2, -1]])


# pandas得到的数据有values属性，是numpy格式
train_features = torch.tensor(all_features.values[:, :-1], dtype=torch.float)
print(train_features.shape)
# test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(all_features.values[:,-1], dtype=torch.int).view(-1, 1)
temp = torch.zeros((train_labels.shape[0], 2), dtype=torch.int)
for i in range(len(train_labels)):
    if train_labels[i][0] == 0:
        temp[i][0] = 1
    else:
        temp[i][1] = 1
train_labels = temp


print(train_labels.shape)

print(train_features[1:5, 1:5])
print(train_labels[-5:-1, :])

# 生成数据集
dataset = Data.TensorDataset(train_features, train_labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
loss = nn.CrossEntropyLoss()



# 返回第i折交叉验证时所需要的训练和验证数据
def get_k_fold_data(k, i, x, y):

    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return x_train, y_train, x_valid, y_valid

def k_fold(k, x_train, y_train, num_epochs,
           lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = nn.Linear(x_train.shape[1], 1)
        train_ls, valid_ls = train(net, *data, num_epochs, lr,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1] # 加入最后一轮的损失
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                     range(1, num_epochs + 1), valid_ls, ['train', 'valid'])

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
net = nn.Linear(train_features.shape[1], 2)
optimizer = optim.SGD(net.parameters(), lr=lr)
def nlog(output, label):
    return -torch.log()
# 训练
def train():
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        for i, (feature, label) in enumerate(data_iter):
            output = net(feature) # 前向传播
            l = loss(output, label) # 计算损失
            optimizer.zero_grad() # 每轮batch_size的梯度都要清零
            l.backward() # 后向传播
            optimizer.step() # 更新参数
            train_l_sum += l.item()  # 计算损失和
            train_acc_sum += output.argmax(dim=1).eq(label.view(-1)).sum().item()
            n += 1
            if (i + 1) % log_interval == 0:
                print('epoch %d, iterator %d, loss: %f' % (epoch, i + 1, loss.item()))
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start))


if __name__ == '__main__':
    train()
