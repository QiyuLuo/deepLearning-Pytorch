import torch
import torch.nn as nn

# 将对象序列化

x = torch.tensor([1, 2, 3])
torch.save(x, "x.txt")

# 反序列化对象

y = torch.load("x.txt")
print(y)