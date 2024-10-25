import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm   # 进度条
import torch
import torch.nn as nn
import torch.optim as optim # 完成梯度下降算法
from data_loader import iris_dataloader

# 初始化神经网络模型

class NN(nn.Module):
    def __init__(self,in_dim,hidden_dim1,hidden_dim2,out_dim) -> None:
        super.__init__()

        # 定义三成模型结构
        self.layer1 = nn.Linear(in_dim,hidden_dim1)  
        self.layer2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2,out_dim)

    '''
    Function: forword函数
    Description: 将结果更新
    Input: x
    Output: None
    Return: void
    param {*} self
    param {*} x
    '''
    def forword(self,x):
        x = self.layer1(x)  #将数据输入进去，使用输出结果覆盖
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        
#定义计算环境,不管电脑是cpu还是gpu都可以运行
device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")

# 训练集 验证集 测试集






