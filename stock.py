import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import math
from matplotlib import pyplot as plt


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

# data cleaning
data_eth = pd.read_csv('ETH.csv', thousands= ',').sort_index(ascending=False)
data_bit = pd.read_csv('BTC.csv', thousands= ',').sort_index(ascending=False)

data_eth.columns = ['Date','Close', 'Open', 'High', 'Low', 'volume', 'variation']
data_eth['volume'] = data_eth['volume'].apply(value_to_float)
data_eth.drop(columns = ['Date', 'variation'], inplace = True)
data_eth.reset_index(drop=True, inplace = True)
data_eth[['Close', 'Open', 'High', 'Low']] = data_eth[['Close', 'Open', 'High', 'Low']].apply(pd.to_numeric)

mid = []
for i in range(len(data_eth)):
    mid.append(int((data_eth.loc[i, 'High'] + data_eth.loc[i, 'Low'])/2))
data_eth['Mid'] = mid


data_bit.columns = ['Date','Close', 'Open', 'High', 'Low', 'volume', 'variation']
data_bit['volume'] = data_bit['volume'].apply(value_to_float)
data_bit.drop(columns = ['Date', 'variation'], inplace = True)
data_bit.reset_index(drop=True, inplace = True)
data_bit[['Close', 'Open', 'High', 'Low']] = data_bit[['Close', 'Open', 'High', 'Low']].apply(pd.to_numeric)

mid = []
for i in range(len(data_bit)):
    mid.append(int((data_bit.loc[i, 'High'] + data_bit.loc[i, 'Low'])/2))
data_bit['Mid'] = mid


#ML\
print(data_eth.shape)
#X = data_eth[['Close', 'Open', 'High', 'Low', 'Mid']]
#Y = data_eth.iloc[:, -1:]
#x_train, x_test, y_train, y_test = train_test_split(X,Y,shuffle= False,test_size = 0.3)
minmax = MinMaxScaler()
x = minmax.fit_transform(data_eth)
print(x[:10])

y = x[:,-1:]
x = x[:,:-1]

#y = minmax.fit_transform(Y)

#test_X = data_bit.iloc[:, :-1]
#test_Y = data_bit.iloc[:, -1:]
#test_x = minmax.fit_transform(x_test)
#test_y = minmax.fit_transform(y_test)
'''
train_x = torch.reshape(train_x_be, (train_x_be.shape[0], 1, train_x_be.shape[1]))
train_y = torch.reshape(train_y_be, (train_y_be.shape[0], 1, train_y_be.shape[1]))
'''
'''
print(dataloader.shape)
'''


'''
train = []
for index in range(len(data_eth) - sequence_length):
    train.append(data_eth[index : index + sequence_length].values) #column 제거
print(train[0:3])

normalized_data = []
norm_data= []
for window in train:
    norm_window = []
    for p in window:
        norm_data = [((float(p[i]) / float(window[0][i])) - 1) for i in range(len(p))]
        norm_window.append(norm_data)
    normalized_data.append(norm_window)


train = np.array(normalized_data)
np.random.shuffle(train)
train = torch.Tensor(train)
print(train.shape)

'''
def timesin(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def make_window(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i : i+window_size])
        label_list.append(label[i+window_size])
    return torch.FloatTensor(np.array(feature_list)) , torch.FloatTensor(np.array(label_list))
class stock_pre(nn.Module):
    def __init__(self,num_classes,input_size, hidden_size, num_layers, seq_length):
        super(stock_pre, self).__init__() # super().__init__
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size = input_size, hidden_size= hidden_size, num_layers= num_layers, batch_first= True)
            #1st layer
        self.fc = nn.Sequential(nn.Linear(hidden_size*seq_length, 1), nn.Tanh()) #matrix multiple. inputsize = layer 입력크기, 64 = layer 출력 크기.(히든 노드 갯수)
        # self.layer2 = nn.Linear(256, 256)
        # self.layer3 = nn.Linear(256, 128)
        # self.layout = nn.Linear(128, num_classes)
        # self.tanh = nn.Tanh()

    def forward(self, x): # 28*28 영상이 64개! 병렬성 증가.
        h_0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(DEVICE)
        c_0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(DEVICE)
        output, _ = self.lstm(x, (h_0,c_0))

        #hn = hn.view(-1, self.hidden_size)
        output = output.reshape(output.shape[0], -1)
        output = self.fc(output)
        # out = self.tanh(hn)
        # out = self.layer1(out)
        # out = self.tanh(out)
        # out = self.layer2(out)
        # out = self.tanh(out)
        # out = self.layer3(out)
        # out = self.tanh(out)
        # out = self.layout(out)

        return output


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE = {}".format(DEVICE))

window_size = 10
x, y = make_window(x, y, window_size)
split = int(len(x)*0.7)


train_x = x[:split]
train_y = y[:split]
test_x = x[split:]
test_y = y[split:]

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)

print(train_x.shape)

dataset = TensorDataset(train_x, train_y)
testset = TensorDataset(test_x, test_y)
dataloader = DataLoader(dataset, batch_size= 20, shuffle= True)
testloader = DataLoader(testset, batch_size= 20, shuffle= False)

input_size = x.size(2)
hidden_size = 8
num_layers = 2
num_classes = y.size(1)

LSTM = stock_pre(num_classes, input_size, hidden_size, num_layers, window_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(LSTM.parameters(), lr=1e-3)  # 요즘엔 adam을 많이 쓰기도 함.
#optimizer = optim.SGD(LSTM.parameters(), lr=1e-3, momentum=0.9)  # 요즘엔 adam을 많이 쓰기도 함.
loss_stack = []

start = time.time()
for epoch in range(10000):
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
        x_train, y_train = data
        optimizer.zero_grad()
        outputs = LSTM.forward(x_train.to(DEVICE))
        loss = criterion(outputs, y_train.to(DEVICE))
        loss.backward()
        optimizer.step()
        running_loss += loss
        if i == 0:
            print('[Epoch %d, step = %5d] loss:%.3f, time = ' % (epoch + 1, i + 1, running_loss), timesin(start))
            loss_stack.append(running_loss)
            running_loss = 0.0
print(timesin(start))

#import matplotlib.ticker as ticker
concatdata = torch.utils.data.ConcatDataset([dataset, testset])
data_loader = torch.utils.data.DataLoader(dataset = concatdata, batch_size= 100)

with torch.no_grad():
    LSTM.eval()
    predictions = torch.tensor([], dtype=torch.float).to(DEVICE)
    real = torch.tensor([], dtype=torch.float).to(DEVICE)
    acc = 0.
    for i, data in enumerate(data_loader, 0):
        ori, target = data
        test_predict = LSTM(ori.to(DEVICE))
        predictions = torch.cat((predictions, test_predict), dim = 0)
        real = torch.cat((real, target.to(DEVICE)), dim = 0)
        loss = criterion(test_predict, target.to(DEVICE))
        acc += torch.sum(test_predict.to(DEVICE) == target.to(DEVICE)).item()
    print('*' * 20, 'Test', '*' * 20)
    print('Loss: {}, Accuracy: {} %'.format(loss.item(), acc / len(test_predict) * 100))
    print('*' * 46)


predictions = predictions.cpu().numpy()
real = real.cpu().numpy()
plt.figure(figsize = (20,10)) # Plotting
plt.title("ETH price predict")
plt.xlabel('period')
plt.plot(real, label = 'Real Data')
plt.plot(predictions, label = 'predicted data')
plt.plot(np.ones(100)*len(dataset),np.linspace(0,1,100),'--',linewidth= 0.6)
plt.legend()
plt.show()

