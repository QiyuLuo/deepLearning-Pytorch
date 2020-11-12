import torch
import torch.nn as nn

# a = torch.randn((3,2,1))
# print(a)
# j = torch.LongTensor([0])
# a0 = a.index_select(0, j)
# a1 = a.index_select(1, j)
# a2 = a.index_select(2, j)
# print(a0, a0.size())
# print(a1, a1.size())
# print(a2, a2.size())
# a = torch.ones(5)
# b = torch.rand((2, 3))
# d = torch.rand((3,1))
# print(torch.mm(b, d)[:,:])
# print(torch.matmul(b, d)[:])
# print((torch.rand((6,5)) < 0.5).float())
# print()
# print(b.view(-1, 3))
# print(b.view((-1, 3)))
# print(a.sum())
# print(type(a.sum()))

def so(a, b, c):
    print(a, b, c)
def func(*args):
    print(args)

def funct(**args):
    print(args)
a = (2, 3, 4)
so(*a)
func(1,2,3)
funct(a = 1, b = 2, c = 4)

