# @File: 含题注的混合图.py
# @Author: chen_song
# @Time: 2025-04-05 15:21

import torch.nn
from torch.nn import PReLU

from utils.smooth import smooth

x = torch.arange(-10.0,10.0,1.0,requires_grad=False)
y1 = torch.relu(x.detach())
y2 = torch.sigmoid(x.detach())
y3 = torch.tanh(x.detach())
# 注意PRelu要求输入的是整数
pRelu = PReLU()
y4 = pRelu(x.detach())

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
y1 = smooth(y1.detach().numpy())[:-1]
y2 = smooth(y2.detach().numpy())[:-1]
y3 = smooth(y3.detach().numpy())[:-1]
y4 = smooth(y4.detach().numpy())[:-1]


plt.plot(x,y1,label="relu",linewidth=3)
plt.plot(x,y2,label="sigmoid",linewidth=3)
plt.plot(x,y3,label="tanh",linewidth=3)
plt.plot(x,y4,label="pRelu",linewidth=3)
plt.title("常见激活函数汇总")

plt.legend(loc="upper left",fontsize=12)
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
plt.show()