import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
import torch.optim as optim
import torch.utils.data
import time
import os
os.path.join('./')
import Data_mining.utils as utils
# 定义超参数

batch_size = 100
num_epochs = 50
log_interval = 10
lr = 0.01

torch.set_default_tensor_type(torch.FloatTensor)
all_features = pd.read_csv('E:/work/data/Android_Data.csv',header=None) # (1500, 887)

print(all_features.shape)
print(type(all_features.values))

# pandas得到的数据有values属性，是numpy格式
train_features = torch.tensor(all_features.values[:, :-1], dtype=torch.float)
train_labels = torch.tensor(all_features.values[:,-1], dtype=torch.long).view(-1)
print(train_features.shape)
print(train_labels.shape)

print(train_features[1:5, 1:5])
print(train_labels[-5:-1])


dataset = Data.TensorDataset(train_features, train_labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 划分数据集，按照比例(tr, va, te)--(7:2:1)
def segementDataset(features, labels, scale = (7, 2, 1)):
    train_size = int(0.7 * features.shape[0])
    val_end = int(0.9 * features.shape[0])
    return (features[0: train_size, :], labels[0: train_size]), (features[train_size: val_end, :], labels[train_size: val_end]), (features[val_end: features.shape[0], :], labels[val_end: features.shape[0]])
train_data, val_data, test_data = segementDataset(train_features, train_labels)
print(train_data[0].size(), train_data[1].size())
# 生成数据集
trainset = Data.TensorDataset(train_data[0], train_data[1])
valset = Data.TensorDataset(val_data[0], val_data[1])
testset = Data.TensorDataset(test_data[0], test_data[1])

train_iter = Data.DataLoader(trainset, batch_size, shuffle=True)
val_iter = Data.DataLoader(valset, batch_size, shuffle=True)
test_iter = Data.DataLoader(testset, batch_size, shuffle=True)

loss = nn.CrossEntropyLoss() # 标签是一维
net = nn.Linear(train_features.shape[1], 2)
optimizer = optim.SGD(net.parameters(), lr=lr)

# 训练
def train(data_iter):
    print('training ..........')
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        total = 0
        for i, (feature, label) in enumerate(data_iter):
            output = net(feature) # 前向传播
            l = loss(output, label) # 计算损失
            optimizer.zero_grad() # 每轮batch_size的梯度都要清零
            l.backward() # 后向传播
            optimizer.step() # 更新参数
            train_l_sum += l.item()  # 计算损失和
            train_acc_sum += output.argmax(dim=1).eq(label.view(-1)).sum().item()
            n += 1
            total += label.shape[0]
            if (i + 1) % log_interval == 0:
                print('epoch %d, iterator %d, loss: %f' % (epoch, i + 1, l.item()))
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch, train_l_sum / n, train_acc_sum / total, time.time() - start))

def test(data_iter):
    print('test ..........')
    epoch = 1
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    start = time.time()
    total = 0
    for i, (feature, label) in enumerate(data_iter):
        output = net(feature) # 前向传播
        l = loss(output, label) # 计算损失
        train_l_sum += l.item()  # 计算损失和
        train_acc_sum += output.argmax(dim=1).eq(label.view(-1)).sum().item()
        n += 1
        total += label.shape[0]
        if (i + 1) % log_interval == 0:
            print('epoch %d, iterator %d, loss: %f' % (epoch, i + 1, l.item()))
    print('epoch %d, loss %.4f, test acc %.3f, time %.1f sec'
          % (epoch, train_l_sum / n, train_acc_sum / total, time.time() - start))
if __name__ == '__main__':
    train(train_iter)
    test(val_iter)
    test(test_iter)


