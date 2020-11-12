import torch
import numpy as np

# 初始化模型参数
num_inputs = 2  # 样本的特征数
num_examples = 1000  # 样本数
true_w = [2, -3.4]  # 生成标签所用的权重
true_b = 4.2  # 生成标签所用的偏差

def generateData():

    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 生成样本
    labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b  # 生成标签
    labels += torch.tensor(np.random.normal(0, 0.01, num_examples), dtype=torch.float32) # 加均值为0，标准差为0.01的正态分布噪声
    return features, labels