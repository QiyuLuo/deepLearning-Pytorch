import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l

batch_size = 256

# torch.Size([60000, 28, 28])
mnist_train = datasets.FashionMNIST(root='../../data', train=True, download=False, transform=transforms.ToTensor())
mnist_test = datasets.FashionMNIST(root='../../data', train=False, download=False, transform=transforms.ToTensor())

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size= batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size= batch_size, shuffle=False, num_workers=num_workers)


print('train len is ', len(mnist_train), 'test len is ', len(mnist_test))

# 可以使用下标访问任意一个样本

feature, label = mnist_train[0]
print(feature.shape, label)

# 将数值标签转换成对应的类别文本标签

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

x, y = [], []
for i in range(10):
    x.append(mnist_train[i][0])
    y.append(mnist_test[i][1])
d2l.show_fashion_mnist(x, get_fashion_mnist_labels(y))

start = time.time()
for x, y in train_iter:
    continue
print('load train data spend %.2f sec' % (time.time() - start))

