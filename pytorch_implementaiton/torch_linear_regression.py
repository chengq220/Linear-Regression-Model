import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_implementaiton.LinearLineDataset import LinearLineDataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y

#manipulate the dataset to fit the requirements of pytorch
#dataset = LinearLineDataset("../linear_line.csv")
#train_loader = DataLoader(dataset,batch_size=6,shuffle=True)

#import the dataset
dataset = pd.read_csv("../linear_line.csv")
x_min,x_max = dataset.iloc[:,0].min(),dataset.iloc[:,0].max()
x = (dataset.iloc[:,0] - x_min)/(x_max-x_min)
x = x.to_numpy()
y = dataset.iloc[:,1].to_numpy()

#convert numpy to tensor
x_t = torch.from_numpy(x).reshape(len(x),1).float()
y_t = torch.from_numpy(y).reshape(len(y),1).float()

#create the dataset loader
tensor_dataset = TensorDataset(x_t, y_t)
train_loader = DataLoader(tensor_dataset,batch_size=5,shuffle=True)

#the linear model with one layer
linear_model = LinearRegression()

#the loss function --> Mean square error
criterion = torch.nn.MSELoss(size_average=False)
#The optimizer is the stochastic gradient descent with learning rate of 0.1
optimizer = torch.optim.SGD(linear_model.parameters(),lr=0.01)

epoch = 300
loss_value = []

for i in range(epoch):
    for x_b,y_b in train_loader:
        # compute the prediction
        y_pred = linear_model(x_b)

        # compute the loss function
        loss = criterion(y_pred, y_b)
        loss_value.append(loss.detach())

        # clear any gradient from the previous pass through gradient descent
        optimizer.zero_grad()

        # perform back propogation and update the weights
        loss.backward()
        optimizer.step()

print(loss_value[len(loss_value)-1])

#make predictions
preds = linear_model(x_t)
print(preds[100])
