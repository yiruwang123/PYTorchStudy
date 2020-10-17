import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#w就是x的系数
w = torch.Tensor([1.0])
# 意思是计算梯度的值并保存
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()   # 反向传播，计算微分
        print('\tgrad', x, y, w.grad.item())
        #这里要给w.data赋值，不能给w赋值，否则grad会消失，或者如下操作，选择下面三行操作的话就不用清除微分了
        # .data就是tensor类型数据的值，不含其他参数
        # w = w - 0.01 * w.grad
        # w = w.data
        # w.requires_grad = True
        w.data = w - 0.01 * w.grad  #  w.data= w.data - 0.01 * w.grad.data
        w.grad.data.zero_()  # 清除w中的微分,不然的话会累加
    print("progress:", epoch, l.item())

print("predict (after ) : ",4 ,forward(4).item())