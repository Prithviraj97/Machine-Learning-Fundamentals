import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    """
    A simple linear regression model.
    Attributes:
        x (array-like): The input feature data.
        y (array-like): The target data.
        slope (float): The slope of the regression line. Default is 0.
        intercept (float): The intercept of the regression line. Default is 0.
        predictions (array-like): The predicted values after fitting the model.
    Methods:
        linearReg():
            Fits the linear regression model to the data.
            Returns the slope, intercept, mean squared error, and predictions.
        predict(x):
            Predicts the target value for a given input feature value.
            Args:
                x (float): The input feature value.
            Returns:
                float: The predicted target value.
        visualize():
            Visualizes the data points and the fitted regression line.
    """
    def __init__(self, x,y, slope=0, intercept=0):
        self.x = x
        self.y = y
        self.slope = slope
        self.intercept = intercept
        self.predictions = None

    def linearReg(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        numerator = np.sum((self.x - x_mean) * (self.y - y_mean))
        denominator = np.sum((self.x - x_mean) ** 2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        self.predictions = self.slope * self.x + self.intercept
        mse = np.mean((self.y - self.predictions) ** 2)
        return self.slope, self.intercept, mse, self.predictions
    
    def predict(self, x):
        price = self.slope * x + self.intercept
        return price
    
    def visualize(self):
        plt.scatter(self.x, self.y, color='red')
        plt.plot(self.x, self.predictions, color='blue')
        plt.show()

#initialize the class
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([300, 350, 500, 700, 800, 850, 900, 900, 1000, 1200])
model = SimpleLinearRegression(x, y)
model.linearReg()
model.visualize()