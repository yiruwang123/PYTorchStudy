import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
#-------------------------------------------------------#

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()
#-------------------------------------------------------#
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#-------------------------------------------------------#
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# [0,10]中取200个点
x = np.linspace(0, 10, 200)
# view就是np中的reshape，转化为200行一列的矩阵，x_t是为了求y的，必须是torch.Tensor类型，不用来画图
x_t = torch.Tensor(x).view((200, 1))
# 用模型求出y
y_t = model(x_t)
# .numpy()就是变成一个np数组
y = y_t.data.numpy()
plt.plot(x, y)
# 画线，就是y=0.5的红线，c是颜色
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()