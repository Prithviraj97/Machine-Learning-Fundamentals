import numpy as np
import matplotlib.pyplot as plt
import math

'''
Our goal here is to first implement a simple linear regression model without using any libraries.
Goal - implement the Linear Regression Algorithm.
       use the linear regression model to predict the price of a house given the area of the house.
       Visualize the data and the model.
'''
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([300, 350, 500, 700, 800, 850, 900, 900, 1000, 1200])

#calculate the mean of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

#calculate the slope and the intercept
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

#make predictions
predictions = slope * x + intercept

#calculate the error
mse = np.mean((y - predictions) ** 2)

#output the results
print(slope, intercept, mse, predictions)

# #visualize the data and the model
# plt.scatter(x, y, color='red')
# plt.plot(x, predictions, color='blue')
# plt.show()
