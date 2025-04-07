# @File: 1.绘制relu激活函数.py
# @Author: chen_song
# @Time: 2025-04-05 13:15

import torch
import torch.nn

x = torch.arange(-10.0,10.0,1.0,requires_grad=True)
y = torch.relu(x)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.plot(x.detach(),y.detach())
plt.title("ReLU激活函数")
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
plt.show()