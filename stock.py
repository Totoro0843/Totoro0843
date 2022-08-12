import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from matplotlib import pyplot as plt
'''
class HelloMLP(nn.Module):
    def __init__(self, input_size = 784, num_classes = 10):
        super(HelloMLP, self).__init__() # super().__init__
        self.mlp = nn.Sequential(
            #1st layer
            nn.Linear(input_size, 64), #matrix multiple. inputsize = layer 입력크기, 64 = layer 출력 크기.(히든 노드 갯수)
            nn.ReLU(), # activation function.

            #2nd layer
            nn.Linear(64, 64), # 두번째 레이어. 히든 아웃풋 64 * 64 weigthed sum.
            nn.ReLU(),

            #3rd (output) layer
            nn.Linear(64, num_classes), # num_claseese = 나가는 갯수.
            ##nn.Softmax(), 보통 쓰는데 여기선 안씀. 왜? 인식이라서. 학습할때는 필요함!
        )

    def forward(self, x): # 28*28 영상이 64개! 병렬성 증가.
        x_ = x.view(x.size(0), -1) # view 함수.
        y_ = self.mlp(x_)
        return y_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE = {}".format(DEVICE))
'''

def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return int(float(x.replace('K', '')) * 1000)
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return int(float(x.replace('M', '')) * 1000000)
        return 1000000.0
    if 'B' in x:
        return int(float(x.replace('B', '')) * 1000000000)
    return 0.0


data_eth = pd.read_csv('ETH.csv', thousands= ',').sort_index(ascending=False)

data_eth.columns = ['Date','Close', 'Open', 'High', 'Low', 'volume', 'variation']
data_eth['volume'] = data_eth['volume'].apply(value_to_float)
data_eth.drop(columns = ['Date', 'variation'], inplace = True)

data_eth.reset_index(drop=True, inplace = True)
print(data_eth.head())

data_eth[['Close', 'Open', 'High', 'Low']] = data_eth[['Close', 'Open', 'High', 'Low']].apply(pd.to_numeric)
print(data_eth.head())
print(data_eth.loc[0, 'High'])

mid = []
for i in range(len(data_eth)):
    mid.append(int((data_eth.loc[i, 'High'] + data_eth.loc[i, 'Low'])/2))
data_eth['Mid'] = mid
print(data_eth.head())

