import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])

w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return w1 * (x ** 2) + w2 * x + b

def loss(x, y):
    y_predict = forward(x)
    return (y_predict - y) ** 2

mse_history = []

for epoch in range(100):
    total_loss = 0
    for x, y in zip(x_data, y_data):
        l = loss(x,y)
        total_loss += l.item()
        l.backward()
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    mse = total_loss / 3.0
    mse_history.append(mse)

print(f"param after 100 epoch: w1 = {w1}, w2 = {w2}, b = {b}")

plt.figure()
plt.subplot(111)
plt.plot(range(len(mse_history)), mse_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Training Epochs')
plt.grid(True)

plt.show()


