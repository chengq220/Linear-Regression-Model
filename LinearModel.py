import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearModel:
    def __init__(self, x):
        # of variables (thetas)
        self.theta = np.random.rand(x) #random weights initialization
        self.alpha = 0.1               #learning rate
        self.regularization = 10       #regularization parameters

    #update the weights of the hypothesis
    def train(self,dataset):
        # Separate the independent and dependent variables and normalizing the dataset
        x = dataset.iloc[:, 0:-1]
        x = (x-x.min())/(x.max()-x.min())

        y = dataset.iloc[:, -1]
        y = (y-y.min())/(y.max()-y.min())

        #add the bias term to the linear regression model
        bias = np.ones_like(x.iloc[:,0]).reshape(len(x.iloc[:,0]),1)
        x.insert(0,"Bias",bias)
        #print(x)

        # One-hot encode the categorical data only for categorical data
        #y_one_hot = self.one_hot_encoding(y)

        # splitting into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

        #train the model
        self.train_model(X_train,Y_train)

        #check the model accuracy with testing set
        self.modelAccuracy(X_test,Y_test)

    def one_hot_encoding(self,y):
        return pd.get_dummies(y)


    def train_model(self, x, y):
        m = len(x)
        epoch = 50
        for i in range(epoch): #running for 25 epochs
        #the loss function with MSE
            loss = np.power(np.subtract(self.predict(x),np.asarray(y)),2)
            loss = np.sum(loss) / (2*m) + (self.regularization / 2 * m) * np.sum(x**2)

            #print("epoch ", i, " loss is ", loss)

            #update the weight by computing the gradient
            self.theta -= (self.alpha/m) * self.gradient_descent(x,y)

    def gradient_descent(self,x,y):
        gradient = np.zeros_like(self.theta)
        a = (np.subtract(self.predict(x), np.asarray(y))).tolist()
        for i in range(len(x.iloc[:,0])):
            b = x.iloc[i,:]
            gradient += (a[i] * b) + (self.regularization/len(x) * x.iloc[i,:])
        return gradient

    def predict(self,input):
        prediction = np.matmul(input,np.transpose(self.theta))
        return prediction

    def modelAccuracy(self,xtest,ytest):
        residue = 0
        y_pred = self.predict(xtest)
        print("The average difference is: ", np.sum(np.abs(y_pred-ytest))/(2*len(y_pred)))

    def predict_discrete(self,input):
        #print(self.theta)
        return np.matmul([1,input],np.transpose(self.theta))

    def __str__(self) -> str:
        return "Theta is: {}".format(self.theta)


