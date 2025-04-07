# @File: 3.conbcate.py
# @Author: chen_song
# @Time: 2025-04-05 13:42
import numpy as np
from sympy.printing.precedence import precedence_Integer

# 1.针对一维数组的拼接
arr1 = np.array([1,2])
arr2 = np.array([3,4])

case1 = np.concatenate((arr1,arr2),axis=0)
print(case1)

# case2 针对2维数组
arr3 = np.array([[1,2],[3,4]])
arr4 = np.array([[5,6],[7,8]])
# 在第一个维度上进行拼接 横着写，总的维度不变
case2 = np.concatenate((arr3,arr4),axis = 0)
print(case2)
# 竖着写 形状不变，总的维度不变
case3 = np.concatenate((arr3,arr4),axis = 1 )
print(case3)

import numpy as np

# 创建三个一维数组
e = np.array([1, 2])
f = np.array([3, 4])
g = np.array([5, 6])

# 拼接三个数组 总的维度不变
result_multiple = np.concatenate((e, f, g),axis=0)
print("多个一维数组拼接结果：", result_multiple)