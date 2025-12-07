import torch
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(data[:,:-1])
y_data = torch.from_numpy(data[:,[-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.elu = torch.nn.ELU()
        self.softplus = torch.nn.Softplus()
        self.hard_tanh = torch.nn.Hardtanh()

    def forward(self, x):
        x = self.hard_tanh(self.linear1(x))
        x = self.hard_tanh(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

loss_history = []

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

print(loss_history)
print('loss:', loss_history[-1])


plt.figure(figsize=(12, 4))
plt.subplot(1, 1, 1)
plt.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('loss vs Training Epochs(hard_tanh)')
plt.grid(True)

plt.tight_layout()
plt.show()