import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(10, 1))
loss = nn.MSELoss()
x = torch.rand(3, 10, requires_grad=True)
y = torch.tensor([1,2,3], dtype=torch.float).view(-1, 1)
net.eval()
output = net(x)
l = loss(output, y)
l.backward()
print(x.grad)


