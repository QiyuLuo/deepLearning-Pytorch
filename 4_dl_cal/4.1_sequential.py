import torch
from collections import OrderedDict
import torch.nn as nn


class mySequential(nn.Module):

    def __init__(self, *args):
        super(mySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果只有一个参数，并且是OrderedDict类型
            for key, module in args[0].items():
                self.add_module(key, module) # add_module方法会将module添加进self._modules(一个OrderedDict)
        else: # 传入一些module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            x = module(x)

        return x

net = mySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
x = torch.rand((3, 784))
output = net(x)
print(output)