import torch
import torch.nn as nn
import torch.nn.init as init
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

x = torch.rand((3, 4))
output = net(x)
print(net)
print(output)
print(x.requires_grad)
print(output.requires_grad)
for name, parm in net.named_parameters():
    if "weight" in name:
        init.normal_(parm, 0, 0.01)
        print(name, parm.size(), parm.data)
    if "bias" in name:
        init.constant_(parm, 0)
        print(name, parm.data)
nn.Parameter
"""

# parm类型是torch.nn.parameter.Parameter,它是tensor的子类。
# 如果在自己的网络中将变量设置为这个类型，那么将会自动加入到网络的参数中
class myModule(nn.Module):
    def __init__(self, **kwargs):
        super(myModule, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand((2, 4)))
        self.weight2 = torch.rand((2, 4))

    def forward(self, x):
        pass

net = myModule()
for name, parm in net.named_parameters():
    print(name)
"""

# 4.共享模型参数,

linear = nn.Linear(2, 2)
net2 = nn.Sequential(linear, linear)
print(net2)
for name, param in net2.named_parameters():
    print(name, param.data)

print(id(net2[0]) == id(net2[1]))
print(id(net2[0].weight) == id(net2[1].weight))
