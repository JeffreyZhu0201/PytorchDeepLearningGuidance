# PytorchDeepLearningGuidance
基于Pytorch的深度学习指南

## 目录
* 神经网络算法
* 感知机模型
* Pytorch API
* 神经网络的有监督训练
* 梯度下降算法
* 

## 神经网络
Wiki百科
> 人工神经网络（英语：artificial neural network，ANNs）又称类神经网络，简称神经网络（neural network，NNs），在机器学习和认知科学领域，是一种模仿生物神经网络（动物的中枢神经系统，特别是大脑）的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统(adaptive system)，通俗地讲就是具备学习功能。
 
NN : 输入层->隐藏层->输入层

Deep Learning = Neural Network

DL : NN CNN RNN Transformer ...

每一层由若干个神经元（节点）组成，每一个节点成为感知机模型



## 感知机模型

f(x1,x2,x3...w1,w2,w3...b) -> y

x : 输入
w : weight,权重
b : bias 偏执

w,b由神经网络训练得到
 

## Pytorch API

CLASS torch.nn.Linear(in_feature,out_feature,bias=True,device=None,dtype=None)

[pytorch官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)



## 神经网络的有监督训练

Data -> Model -> Result <- Distance, loss fuction -> Label

训练减少Distance <=> 求损失函数最小值

loss = 1/2(y-y0)^2

## 梯度下降算法

梯度 ~ 导数
θ = θ1 - lr*g
(g为梯度，lr为学习速率)

θ（w）函数为损失函数
将损失减少到最小

## 神经网络训练API
1. DataLoader
2.

```python
loss = nn.CrossEntropyLoss()
loss.backward()
optimizer.step()
```
