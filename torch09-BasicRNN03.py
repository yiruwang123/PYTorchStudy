
import torch

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
#𝒔𝒆𝒒𝑳𝒆𝒏, 𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 𝒉𝒊𝒅𝒅𝒆𝒏𝑺𝒊𝒛e
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
#𝒔𝒆𝒒𝑳𝒆𝒏 × 𝒃𝒂𝒕𝒄𝒉𝑺𝒊𝒛𝒆, 1
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)
    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))