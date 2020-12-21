import torch
import pandas as pd
import torch.utils.data

file = 'E:/work/data/Android_Data.csv'
# 从csv文件生成数据集，没有列名称，最后一列为类别,返回样本和标签
def generateDataSet(file):
    torch.set_default_tensor_type(torch.FloatTensor)
    all_features = pd.read_csv(file, header=None)  # (1500, 887)
    # pandas得到的数据有values属性，是numpy格式
    train_features = torch.tensor(all_features.values[:, :-1], dtype=torch.float)
    train_labels = torch.tensor(all_features.values[:, -1], dtype=torch.long).view(-1)
    return train_features, train_labels

# 将数据集分割为训练集，验证集，测试集。
def segementDataset(features, labels, scale = torch.tensor([7, 2, 1])):
    total = scale.sum().float()
    train_size = int(scale[0] / total * features.shape[0])
    val_end = int((scale[0] + scale[1]) / total * features.shape[0])
    return (features[0: train_size, :], labels[0: train_size]), (features[train_size: val_end, :], labels[train_size: val_end]), \
           (features[val_end: features.shape[0], :], labels[val_end: features.shape[0]])