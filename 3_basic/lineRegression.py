# from matplotlib import pyplot as plt
from generateDataset import *
from d2lzh_pytorch import *

# 定义超参数
lr = 0.03
num_epochs = 1
net = linreg
batch_size = 10
log_interval = 10 # 打印日志间隔


# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)



features, labels = generateData()

# 绘制第二个特征features[:, 1]和标签 labels 的散点图
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()

# 训练模型
print('init weight is ', w)
def train():
    for epoch in range(num_epochs):
        i = 0
        for feature, label in data_iter(batch_size, features, labels):
            output = net(feature, w, b) # 前向传播得到预测
            loss = squared_loss(output, label).sum() # 根据预测和标签得到一个batch_size的损失
            loss.backward() # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size) # 优化算法，更新权重和偏差

            # 每次迭代需要将梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
            i += 1
            if i % log_interval == 0:
                print('epoch %d, iterators %d loss %f' % (epoch + 1, i * log_interval, loss.mean().item()))

        # 训练完毕，打印学习到的参数和设定的真实参数进行比较
        print('epoch ', (epoch + 1), 'predict weight is ', w)
        print('epoch ', (epoch + 1), 'true weight is ', true_w)
        print('epoch ', (epoch + 1), 'predict bias is ', b)
        print('epoch ', (epoch + 1), 'true bias is ', true_b)

if __name__ == '__main__':
    train()