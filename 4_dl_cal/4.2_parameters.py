import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

x = torch.rand((3, 4))
output = net(x)
print(net)
print(output)
print(x.requires_grad)
print(output.requires_grad)
for name, parm in net.named_parameters():
    print(name, parm.size(), parm)