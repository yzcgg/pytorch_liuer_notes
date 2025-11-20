# task: try to use this model (y = a * x + b)
# and draw the cost graph

# 假设学生成绩 和 学习时长 + 平时成绩 有关
# 服从：y = 2 * x + 10
# 训练集有4组（x，y）
# 1,12
# 2,14
# 3,16
# 4,18

# import numpy as np
# # import matplotlib.pyplot as plt
#
# x_data = [1.0, 2.0, 3.0, 4.0]
# y_data = [12.0, 14.0, 16.0, 18.0]
#
# def forward(x):
#     return a * x + b
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y - y_pred) ** 2
#
# a_list = []
# b_list = []
# mse_list = []
# for a in np.arange(0.0, 4.1, 0.1):
#     for b in np.arange(8.0, 12.1, 0.1):
#         print('a = ', a, ', b = ', b)
#         l_sum = 0
#         for x_val, y_val in zip(x_data, y_data):
#             l_sum += loss(x_val, y_val)
#         mse = l_sum / len(x_data)
#         print('MSE = ', mse)
#         a_list.append(a)
#         b_list.append(b)
#         mse_list.append(mse)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据准备
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [12.0, 14.0, 16.0, 18.0]

def forward(x, a, b):
    return a * x + b

def loss(x, y, a, b):
    y_pred = forward(x, a, b)
    return (y - y_pred) ** 2

# 创建参数网格
a_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(8.0, 12.1, 0.1)
A, B = np.meshgrid(a_range, b_range)

# 计算每个(a,b)组合的MSE
MSE = np.zeros_like(A)
for i in range(len(a_range)):
    for j in range(len(b_range)):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, A[j,i], B[j,i])
        MSE[j,i] = l_sum / len(x_data)

# 创建交互式图形
plt.ion()  # 开启交互模式
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(A, B, MSE, cmap='viridis', alpha=0.8)

# 标记最小值点和真实参数点
min_idx = np.unravel_index(np.argmin(MSE), MSE.shape)
min_a = A[min_idx]
min_b = B[min_idx]
min_mse = MSE[min_idx]

ax.scatter(min_a, min_b, min_mse, color='red', s=100, label=f'Minimum (a={min_a:.1f}, b={min_b:.1f})')

true_a, true_b = 2.0, 10.0
true_mse = 0
ax.scatter(true_a, true_b, true_mse, color='green', s=100, label='True parameters (a=2.0, b=10.0)')

# 设置坐标轴标签和标题
ax.set_xlabel('Parameter a')
ax.set_ylabel('Parameter b')
ax.set_zlabel('MSE Loss')
ax.set_title('3D Cost Function Surface: MSE vs Parameters (a, b)\n'
             'Use mouse to rotate, scroll to zoom')

# 添加颜色条和图例
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.legend()

# 显示图形并保持交互
plt.show()

# 在控制台显示操作提示
print("\n=== 交互式三维图形操作指南 ===")
print("1. 使用鼠标左键拖动：旋转视角")
print("2. 使用鼠标右键拖动：平移视图")
print("3. 使用滚轮：缩放视图")
print("4. 关闭图形窗口：退出程序")

# 保持程序运行，直到图形窗口关闭
plt.ioff()  # 关闭交互模式，但保持图形显示
plt.show()