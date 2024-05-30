import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import time

# 创建一个简单的RNN
rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)

# 创建一些随机输入数据
input_data = torch.randn(1, 5, 1)
target_data = torch.randn(1, 10)

# 初始化图像
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

def update_plot(hidden_np):
    ax.clear()
    ax.imshow(hidden_np[0], cmap='hot', interpolation='nearest')
    plt.draw()
    plt.pause(0.01)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

for i in range(1000):
    # 通过RNN运行数据
    output, hidden = rnn(input_data)

    # 计算损失
    loss = criterion(output.view(-1, 10), target_data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每10个迭代更新一次图像
    if i % 10 == 0:
        # 将隐藏状态转换为numpy数组
        hidden_np = hidden.data.numpy()

        # 更新图像
        update_plot(hidden_np)

    # 模拟训练过程
    time.sleep(0.1)

plt.ioff()
plt.show()