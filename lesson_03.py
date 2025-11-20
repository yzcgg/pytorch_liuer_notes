import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

w = torch.Tensor([1.0])  # 梯度下降，求的就是w在什么时候，让loss最小
w.requires_grad = True

# Lists to store values for plotting
w_history = []
mse_history = []

def forward(x):
    return x * w

def loss(x, y):
    return (forward(x) - y) ** 2

# train
for epoch in range(100):
    total_loss = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        total_loss += l.item()
        l.backward()  # 会释放计算图
        w.data = w.data - 0.01 * w.grad.data # 梯度下降
        print("\tx = ", x," y = ", y," grad = ", w.grad.item(), " w = ", w.item())
        w.grad.data.zero_()

    avg_loss = total_loss / 4.0
    w_history.append(w.item())
    mse_history.append(avg_loss)

    #  8.70010994731274e-05 这个数字是 8.7 * 10的-5次方，MSE确实随着训练轮数越来越小。
    print("progress:", epoch, " MSE:", avg_loss)

print(f"Final w value: {w.item()}")

# Plotting the relationship between w and MSE
plt.figure(figsize=(12, 4))

# Plot 1: MSE vs Epochs
plt.subplot(1, 2, 1)
plt.plot(range(len(mse_history)), mse_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Training Epochs')
plt.grid(True)

# Plot 2: w vs MSE
plt.subplot(1, 2, 2)
plt.plot(w_history, mse_history, 'ro-', markersize=4, linewidth=1)
plt.xlabel('w value')
plt.ylabel('MSE')
plt.title('Relationship between w and MSE')
plt.grid(True)

plt.tight_layout()
plt.show()