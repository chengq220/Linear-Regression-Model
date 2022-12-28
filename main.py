import pandas as pd
import matplotlib.pyplot as plt
from LinearModel import LinearModel

#Read from the iris dataset
dataset = pd.read_csv("linear_line.csv")

#visualizing the iris dataset
plt.plot(dataset.iloc[:,0],dataset.iloc[:,1])
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

#Linear model with two weights
model = LinearModel(2)
model.train(dataset)


