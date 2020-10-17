import numpy as np
import torch

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])    # np转为tensor，不要最后一列
# y_data后面的-1要用[],因为必须为矩阵不能为向量
y_data = torch.from_numpy(xy[:, [-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear1 = torch.nn.Linear(8, 6)
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 1)
#         self.activate = torch.nn.ReLU()
#     def forward(self, x):
#         x = self.activate(self.linear1(x))
#         x = self.activate(self.linear2(x))
#         x = torch.sigmoid(self.linear3(x))
#         return x

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for epoch in range(100):
    # Forward，这里没有用到mini-batch
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()