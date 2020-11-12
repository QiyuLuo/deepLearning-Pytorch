import torch
import time
import numpy as np
size = int(1e6)
print(size)
a = torch.ones(size)
b = torch.ones(size)

start = time.time()

c = torch.empty(size)
x = np.ones(size)
y = np.ones(size)
d = np.empty(size)
# numpy- 标量加
for i in range(len(x)):
    d[i] = x[i] + y[i]

# torch
# for i in range(len(a)):
#     c[i] = a[i] + b[i]
print('scalar compute time is ', time.time() - start)

start = time.time()

# d = a + b
# 矢量加
d = x + y
print('vector compute time is {:.20f}'.format(time.time() - start) )

