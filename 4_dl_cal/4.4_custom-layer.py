import torch
import torch.nn as nn

# 自定义一个layer

class selfLayer(nn.Module):
    def __init__(self, **kwargs):
        super(selfLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

net = selfLayer()

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, requires_grad=True)
y = net(x)
print(y)
y = y.sum()
y.backward()
print(x.grad)

net = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
# state_dict 将参数名称映射到参数Tensor的字典对象， 只有可学习参数的层（线性层，卷积层等）才有state_dict条目
print(net.state_dict())

# 包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr= 0.0001, momentum=0.9)
print(optimizer.state_dict())