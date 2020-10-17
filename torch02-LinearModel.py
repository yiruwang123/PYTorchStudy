import torch
import time
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# 模型类必须继承自 nn.Module，类后面括号这是py的继承方式
# backward会由Module自动完成
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # torch.nn.Linear是一个类,参数是输入和输出的feature数，即列数
        self.linear = torch.nn.Linear(1, 1)   # 构造一个Linear对象

    # Class nn.Linear has implemented the magic method __call__(), which enable
    # the instance of the class can be called just like a function. Normally the
    # forward() will be called.
    # forward要被覆写
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# loss函数，size_average表示是不是要求平均，设不设都行
criterion = torch.nn.MSELoss(reduction='sum')

# 优化器，model.parameters()是找权重的，即w，b；lr-learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

starttime = time.time()
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()  #梯度归零之后进行反向传播，不然的话会累加，也可以放在step后面
    loss.backward()
    optimizer.step()
endtime = time.time()
# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

print('time = ', endtime - starttime)