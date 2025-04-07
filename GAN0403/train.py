import torch
import torchvision.utils
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import Discriminator, Generator

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 定义按照imageNet一样的标准化，因为有三个通道，所有均值以及方差各有三个值
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # 从指定文件夹读取图片，然后按照指定方式进行预处理然后返回
    trainset = datasets.ImageFolder(r'E:\pycharmProjects\GAN0403\cartoon\FACE', data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

    def imshow(inputs, picname):
        # 正确的反标准化操作
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        inputs = inputs * std + mean
        inputs = inputs.numpy().transpose((1, 2, 0))
        plt.imshow(inputs)
        plt.savefig(picname + ".jpg")
        # 去掉 plt.close() 和 plt.ion()

    inputs, _ = next(iter(trainloader))
    imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

    # 初始化生成器以及判别器
    d = Discriminator(3,32)
    g = Generator(3,128,1024,100)
    # 由于鉴别器是一个二元分类器,这里定义一个二元交叉熵损失函数
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(d.parameters(),lr = 0.0003)
    g_optimizer = torch.optim.Adam(g.parameters(),lr = 0.0003)


    def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=1000, print_every=10):
        iter_count = 0
        for epoch in tqdm(range(epochs)):
            for inputs, _ in trainloader:
                real_inputs = inputs
                fake_inputs = g(torch.randn(1,100))
                real_labels = torch.ones(real_inputs.size(0))
                # 由于只有一张图片，所以标签只有一个0
                fake_labels = torch.zeros(1)
                # 根据实际输入获取实际输出
                real_outputs = d(real_inputs)
                # 真实数据上判别器损失
                d_loss_real = criterion(real_outputs.squeeze(0),real_labels)

                fake_outputs = d(fake_inputs)
                # 虚假数据上判别器损失
                d_loss_fake = criterion(fake_outputs.squeeze(0),fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                fake_inputs = g(torch.randn(1,100))
                outputs = d(fake_inputs)
                real_labels = torch.ones(outputs.size(0))
                g_loss = criterion(outputs.squeeze(0),real_labels)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                if iter_count % show_every == 0:
                  print('Epoch:{},Iter:{},D:{:.4f},G:{:.4f}'.format(epoch,iter_count,d_loss.item(),g_loss.item()))
                  picname = "Epoch_"+str(epoch)+"Iter_"+str(iter_count)
                  imshow(torchvision.utils.make_grid(fake_inputs.data),picname)

                if iter_count % print_every == 0:
                 print('Epoch:{},Iter:{},D:{:.4f},G:{:.4f}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
                 iter_count += 1

        print("训练完毕~")
    train(d,g,criterion,d_optimizer,g_optimizer,epochs=100)


