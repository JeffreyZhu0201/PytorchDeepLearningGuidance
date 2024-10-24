'''
Author: JeffreyZhu 1624410543@qq.com
Date: 2024-10-24 17:00:24
LastEditors: JeffreyZhu 1624410543@qq.com
LastEditTime: 2024-10-24 18:07:51
FilePath: \PytorchDeepLearningGuidance\data_loader.py
Description: File Description Here...

Copyright (c) 2024 by JeffreyZhu, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy

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

        df = pd.read_csv(self.data_path,name=[0,1,2,3,4]) # 将数据集文件读取到

        d = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}

        df[4] = df[4].map(d)
        
        data = df.iloc[:,:4]
        label = df.iloc[:,4:]
    


