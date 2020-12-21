import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
import torch.optim as optim
import time
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('E:/data/kaggle_house/train.csv') # (1460, 81)
test_data = pd.read_csv('e:/data/kaggle_house/test.csv') # (1459, 80)
n_train = train_data.shape[0]
k = 5
num_epochs = 10
lr = 5
weight_decay = 0
batch_size = 64
log_interval = 10
# 原始数据的第一个特征是id,不能作为特征来推断测试集所以总特征只有79个，最后一个是价格

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
"""
MSSubClass       int64
MSZoning        object
LotFrontage    float64
LotArea          int64
Street          object
"""
# 将连续数值特征标准化，每个值减去μ在除以标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(all_features[:n_train].values[0:5, 0:5])
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)

all_features[numeric_features] = all_features[numeric_features].fillna(0)


all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape) # (2919, 331) 特征从79->331
print(all_features.iloc[1459:1465, -4:-1])
# pandas得到的数据有values属性，是numpy格式

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
train_dataset = Data.TensorDataset(train_features, train_labels)
data_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(train_features[-5:-1, -4:-1])
loss = nn.MSELoss()
net = nn.Linear(train_features.shape[1], 1)
optimizer = optim.SGD(net.parameters(), lr=lr)
def log_rmse(output, label):
    # with torch.no_grad():
    # 将小于1的值设成1，使得取对数时数值更稳定
    pred = torch.max(output, torch.tensor(1.0))
    rmse = torch.sqrt(loss(pred.log().view(-1, 1), label.log().view(-1, 1)))
    return rmse

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for feature, label in data_iter:
            output = net(feature)
            optimizer.zero_grad()
            l = log_rmse(output, label)
            # l = log_rmse(net, feature, label)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_ls.append(log_rmse(net(train_features), train_labels))
            if test_features is not None:
                test_ls.append(log_rmse(net(test_features), test_labels))
        if epoch % 10 == 0:
            print('train rmse %f, valid rmse %f' % (train_ls[-1], test_ls[-1]))
            # print('train rmse %f' % (train_ls[-1]))
    return train_ls, test_ls


# 训练
def train(data_iter):
    print('training ..........')
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        total = 0
        for i, (feature, label) in enumerate(data_iter):
            output = net(feature) # 前向传播
            l = log_rmse(output, label) # 计算损失
            optimizer.zero_grad() # 每轮batch_size的梯度都要清零
            l.backward() # 后向传播
            optimizer.step() # 更新参数
            train_l_sum += l.item()  # 计算损失和
            n += 1
            total += label.shape[0]
            # if (i + 1) % log_interval == 0:
            #     print('epoch %d, iterator %d, loss: %f' % (epoch, i + 1, l.item()))
        if epoch % 10 == 0:
            print('epoch %d, loss %.4f, time %.1f sec'
                  % (epoch, train_l_sum / n, time.time() - start))

if __name__ == '__main__':
    train(data_iter)
    # print('%d-fold validation: avg train rmse %f, avg valid rmse' % (k, train_ls))
