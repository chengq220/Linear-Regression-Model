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

#import the dataset and manipulte it to fit the requirements of pytorch
dataset = LinearLineDataset("../linear_line.csv")

#test-train split
train_size = int(0.8 * dataset.__len__())
test_size = int(dataset.__len__() - train_size)
train_dataset,test_dataset = torch.utils.data.random_split(dataset.__getdataset__(), [train_size, test_size])

train_loader = DataLoader(train_dataset,batch_size=6,shuffle=True)
test_loader = DataLoader(test_dataset)

#the linear model with one layer
linear_model = LinearRegression()

#the loss function --> Mean square error
criterion = torch.nn.MSELoss(size_average=False)
#The optimizer is the stochastic gradient descent with learning rate of 0.1
optimizer = torch.optim.SGD(linear_model.parameters(),lr=0.01)

epoch = 300
loss_value = []

for i in range(epoch):
    for i, (x_b,y_b) in enumerate(train_loader):
        # compute the prediction
        y_pred = linear_model(x_b)

        # compute the loss function
        loss = criterion(y_pred, y_b)

        if i%10==0:
            print("loss at this step is", loss)
        loss_value.append(loss.detach())

        # clear any gradient from the previous pass through gradient descent
        optimizer.zero_grad()

        # perform back propogation and update the weights
        loss.backward()
        optimizer.step()

#test-cases validation
for i, (x,y) in enumerate(test_loader):
    y_pred = linear_model(x)
    print("y: ", y, " y_pred:", y_pred)

#normalize the custom prediction data
value = (303.0 - 1.0)/ (300.0 - 1.0)
new_var = Variable(torch.Tensor([[value]]))
pred_y = linear_model(new_var)
print("predict (after training)", 303, linear_model(new_var).item())
