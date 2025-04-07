# @File: dataloader.py
# @Author: chen_song
# @Time: 2025-04-03 19:37
import cv2
# 加载目标数据
# 导入数据集加载的库
import torchvision.datasets as td
# 导入pytorch中计算机视觉库
import torchvision.transforms as tf
from torch.utils.data import DataLoader,Dataset
# 1.自定义数据集,预留了一个变量进行后期加载数据集的时候进行图像的预处理操作preprocess
class customDataSet(Dataset):
  def __init__(self,imgPath,transform=None):
      self.imgPath = imgPath
      self.transform = transform
  def __len__(self):
      return len(self.imgPath)
  def __getitem__(self, idx):

      img_path = self.imgPath[idx]
      targetImg = cv2.imread(img_path)
      targetImg = cv2.cvtColor(targetImg,cv2.COLOR_BGR2RGB)
      if self.transform:
        targetImg = self.transform(targetImg)
      return targetImg
