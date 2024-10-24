'''
Author: JeffreyZhu 1624410543@qq.com
Date: 2024-10-24 17:00:24
LastEditors: JeffreyZhu 1624410543@qq.com
LastEditTime: 2024-10-24 18:27:16
FilePath: \PytorchDeepLearningGuidance\data_loader.py
Description: File Description Here...

Copyright (c) 2024 by JeffreyZhu, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


'''
Class: iris_dataloader
Description: 自定义导入数据集
Input: 
@params: data_path 数据集文件
Output: None
Return: void
'''
class iris_dataloader(Dataset):
    def __init__(self,data_path):# 构造函数
        self.data_path = data_path  # 复制参数

        assert os.path.exists(self.data_path), "dataset does not found" # 判断是否存在数据集文件

        df = pd.read_csv(self.data_path,name=[0,1,2,3,4]) # 将数据集文件读取到数组 name为列打上标签0，1，2...

        d = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}    # 名称转换字典

        df[4] = df[4].map(d)    # 将第四列花名转换为数值
        
        data = df.iloc[:,:4]    # 取出前四列 : 数据
        label = df.iloc[:,4:]   #取出第五列 : 标签

        data = (data - np.mean(data))/np.std(data) # 数据规范化操作，为了增加训练稳定性，z值化
        # (data - data的平均值)/方差

        self.data = torch.from_numpy(np.array(data,dtype="float32")) 
        self.label = torch.from_numpy(np.array(label,dtype="int64"))
        # 将float32,int类型转换为torch需要的Tensor类型
        
        self.data_num = len(data)
        print("The length of dataset is " + self.data_num)
    



