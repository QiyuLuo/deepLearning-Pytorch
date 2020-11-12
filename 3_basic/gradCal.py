import torch
import numpy as np
a = torch.ones(2, 2, requires_grad= True)
print(a[:,0])
b = a + 2
print('b requires_grad = ', b.requires_grad)
print(b.grad_fn)
z = b * b * 3
out = z.mean()
print('out = ',out)
out.backward()
print('b grad = ',b.retain_grad())
print('a grad = ', a.grad)
print(a.data)
a = torch.ones(2)
print(a)

b = torch.rand(2)
print(b)

b = b.view(2, -1)
print(b)
bias = torch.tensor(np.random.normal(0, 0.01, size=2))
b += bias
print('bias = ', bias)
print('b = ', b)
# print(list(map(lambda x, y: x * y, a, b)))

