import torch
import numpy as np
from pytorch_implementaiton.LinearLineDataset import LinearLineDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y

#manipulate the dataset to fit the requirements of pytorch
linedataset = LinearLineDataset("../linear_line.csv")
train_loader = DataLoader(linedataset,batch_size=5,shuffle=True)

#the linear model with one layer

linear_model = LinearRegression()

#the loss function --> Mean square error
criterion = torch.nn.MSELoss(size_average=False)

#The optimizer is the mini-batch gradient descent with learning rate of 0.1
#calculate the gradient for each batch and find the mean and then update the weight
optimizer = torch.optim.SGD(linear_model.parameters(),lr=0.01)

epoch = 15
loss_value = []

for i in range(epoch):
    for batch in train_loader:
        x = batch[:,0].reshape(-1,1)
        y = batch[:,1].reshape(-1,1)

        # compute the prediction
        y_pred = linear_model(x)
        print(y_pred)

        # compute the loss function
        loss = criterion(y, y_pred)
        loss_value.append(loss.detach())

        # clear any gradient from the previous pass through gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#a line showing the loss function over each iteration of the batch gradient descent
iteration = np.arange(0,len(loss_value))
plt.figure()
plt.plot(iteration,loss_value)
plt.xlabel("iteration")
plt.ylabel("loss")

plt.show()

new_var = Variable(torch.Tensor([[100]]))
pred_y = linear_model(new_var)
print("predict (after training)", 100, pred_y.detach())