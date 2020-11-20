import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data
import sys
import d2lzh_pytorch as d2l

torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('E:/data/kaggle_house/train.csv') # (1460, 81)
test_data = pd.read_csv('e:/data/kaggle_house/test.csv') # (1459, 80)

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.2, 0, 64

# 原始数据的第一个特征是id,不能作为特征来推断测试集所以总特征只有79个，最后一个是价格
# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(len(all_features.dtypes[all_features.dtypes == 'object'].index))
"""
MSSubClass       int64
MSZoning        object
LotFrontage    float64
LotArea          int64
Street          object
"""
# 将连续数值特征标准化，每个值减去μ在除以标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)
# print(all_features.iloc[8:13, [1, 2, 3, 4, 5, 6, -3, -2, -1]])

# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转成指示特征，将那些dtype是object的转换
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape) # (2919, 331) 特征从79->331
# print(all_features.iloc[8:13, [1, 2, 3, 4, 5, 6, -3, -2, -1]])
n_train = train_data.shape[0]
# pandas得到的数据有values属性，是numpy格式
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

loss = nn.MSELoss()



class regNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(regNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        # for param in self.linear.parameters():
        #     nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x):
        return self.linear(x)

def log_rmse(output, label):
    # with torch.no_grad():
    # 将小于1的值设成1，使得取对数时数值更稳定
    pred = torch.max(output, torch.tensor(1.0))
    rmse = torch.sqrt(loss(pred.log().view(-1, 1), label.log().view(-1, 1)))
    return rmse

# def regNet(num_inputs, num_outputs):
#     net = nn.Linear(num_inputs, num_outputs)
#     for param in net.parameters():
#         nn.init.normal_(param, mean=0, std=0.01)
#     return net
# def log_rmse(net, features, labels):
#     with torch.no_grad():
#         # 将小于1的值设成1，使得取对数时数值更稳定
#         clipped_preds = torch.max(net(features), torch.tensor(1.0))
#         rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
#     return rmse.item()
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
        net = regNet(x_train.shape[1], 1)
        train_ls, valid_ls = train(net, *data, num_epochs, lr,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1] # 加入最后一轮的损失
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                     range(1, num_epochs + 1), valid_ls, ['train', 'valid'])

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

if __name__ == '__main__':

    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # net = regNet(train_features.shape[1], 1)
    # train_ls, valid_ls = train(net, train_features, train_labels, None, None, num_epochs, lr,
    #                            weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
    # print('%d-fold validation: avg train rmse %f, avg valid rmse' % (k, train_ls))
