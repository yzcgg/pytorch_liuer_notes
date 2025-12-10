import torch

x_data = torch.Tensor([
    [1.0],
    [2.0],
    [3.0]
])
y_data = torch.Tensor([
    [2.0],
    [4.0],
    [6.0]
])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()