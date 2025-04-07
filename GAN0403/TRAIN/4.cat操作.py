# @File: 4.cat操作.py
# @Author: chen_song
# @Time: 2025-04-05 14:11
import torch

tensor1 = torch.tensor([1,2])
tensor2 = torch.tensor([3,4])
# case1 一维张量拼接
case1 = torch.cat((tensor1,tensor2),dim=0)
print(case1)
# case2 二维张量拼接
tensor3 = torch.tensor([[1,2],[3,4]])
tensor4 = torch.tensor([[3,4],[5,6]])
case2 = torch.cat((tensor3,tensor4),dim=0)
print(case2)
case3 = torch.cat((tensor3,tensor4),dim=1)
print(case3)
# 三维张量拼接
tensor5 = torch.tensor([[1,2],[3,4]])
tensor6 = torch.tensor([[3,4],[5,6]])
tensor7 = torch.tensor([[3,4],[5,6]])

case4 = torch.cat((tensor5,tensor6,tensor7),dim=-1)
print(case4)