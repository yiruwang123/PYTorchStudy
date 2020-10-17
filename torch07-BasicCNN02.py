# padding
# stride
import torch

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]

input = torch.Tensor(input).view(1, 1, 5, 5)

# 构造CNN时加入padding，扩大output维度，这里不需要bias
#conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

# 使用stride，stride是卷积时中心每次移动的距离，有利于降低output维度
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)

kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)

conv_layer.weight.data = kernel.data

output = conv_layer(input)

print(output)
