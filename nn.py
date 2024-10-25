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

# 训练集 验证集 测试集 -> 反应模型的性能
# 导入训练集数据
custom_dataset = iris_dataloader("./PytorchDeepLearningGuidance/")

train_size = int(len(custom_dataset) * 0.7)                  #定义训练集数据长度
val_size = int(len(custom_dataset) * 0.2)                    #定义验证集数据长度
test_size = int(len(custom_dataset) - train_size - val_size) #定义验证集数据长度

# 将数据集按长度分割
train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(custom_dataset,[train_size,val_size,test_size])

# 批量封装，数据集划分和加载
# batch_size:定义一次训练大小，批量大小;shuffle:是否下次批量训练时是否打散
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True) 
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False) 
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False) 

print("训练集大小: ",len(train_loader)*16,"验证集大小: ",len(val_loader),"测试集大小: ",len(test_loader))


# 定义一个推理函数，计算并返回准确率
def infer(model,dataset,device):
    model.eval() # 将模型转到验证状态
    acc_num = 0 # 模型预测正确数量
    # 上下文管理器，避免改变模型参数
    with torch.no_grad():
        for data in dataset:
            datas,labels = data

            out_put = model(datas.to(device)) # 传进设备进行训练，返回二维张量

            predict_y = torch.max(out_put,dim=1)[1] # 把最大可能性的识别出来
            # 第一个维度为数量，第二个维度为结果
            acc_num += torch.eq(predict_y,labels.to(device)).sum().item()
            #数据为批量数据，要对所有的数据结果进行累加，表示当前模型预测正确的数量

    acc = acc_num/len(dataset)
    return acc


# 训练过程
def main(lr = 0.005,epochs = 20):
    model = NN(4,12,6,3).to(device) # 实例化模型
    loss_f = nn.CrossEntropyLoss() # 交叉熵损失函数

    pg = [p for p in model.parameters() if p.requires_grad]
    # p 为所有可训练参数

    


