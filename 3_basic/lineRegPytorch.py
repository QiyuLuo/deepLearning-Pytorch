import torch
from generateDataset import *
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
# 定义超参数

batch_size = 10
num_epochs = 3
log_interval = 10

# 生成数据集
features, labels = generateData()
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型

class LinearNet(nn.Module):

    # 输入网络的特征数
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # 前向传播
    def forward(self, x):
        return self.linear(x)

net = LinearNet(num_inputs)
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))
print(net) # 打印网络的结构
"""
用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])
"""

# 打印可学习的参数

for param in net.parameters():
    print(param)

# 初始化模型参数
# net[0]这样根据下标访问子模块的写法只有当net是个ModuleList或者Sequential实例时才可以
# nn.init.normal_(net.linear.weight, mean=0, std=0.01)
# nn.init.constant_(net.linear.bias, val=0)
"""
注：如果这里的net是用3.3.3节一开始的代码自定义的，那么上面代码会报错，
net[0].weight应改为net.linear.weight，bias亦然。
因为net[0]这样根据下标访问子模块的写法只有当net是个ModuleList或者Sequential实例时才可以，详见4.1节。
"""

# 定义损失函数
# loss = nn.MSELoss()
mseloss = nn.MSELoss()
# torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""
我们还可以为不同子网络设置不同的学习率，这在finetune时经常用到。例：

optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
"""
"""
有时候我们不想让学习率固定成一个常数，那如何调整学习率呢？主要有两种做法。
一种是修改optimizer.param_groups中对应的学习率，
另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，
故而可以构建新的optimizer。但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，
可能会造成损失函数的收敛出现震荡等情况。
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
"""

# 训练
def train():
    for epoch in range(1, num_epochs + 1):

        for i, (feature, label) in enumerate(data_iter):
            output = net(feature) # 前向传播
            loss = mseloss(output, label.view(-1, 1)) # 计算损失
            optimizer.zero_grad() # 每轮batch_size的梯度都要清零
            loss.backward() # 后向传播
            optimizer.step() # 更新参数
            if (i + 1) % log_interval == 0:
                print('epoch %d, iterator %d, loss: %f' % (epoch, i + 1, loss.item()))

        dense = net.linear
        # 训练完毕，打印学习到的参数和设定的真实参数进行比较
        print('epoch ', epoch, 'predict weight is ', dense.weight)
        print('epoch ', epoch, 'true weight is ', true_w)
        print('epoch ', epoch, 'predict bias is ', dense.bias)
        print('epoch ', epoch, 'true bias is ', true_b)

if __name__ == '__main__':
    train()


