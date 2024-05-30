import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# 创建一个简单的RNN
rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)

# 创建一些随机输入数据
input_data = torch.randn(1, 5, 1)

# 通过RNN运行数据
output, hidden = rnn(input_data)

# 将隐藏状态转换为numpy数组
hidden_np = hidden.data.numpy()

# 绘制隐藏状态的热图
plt.figure(figsize=(10, 5))
plt.imshow(hidden_np[0], cmap='hot', interpolation='nearest')
plt.show()