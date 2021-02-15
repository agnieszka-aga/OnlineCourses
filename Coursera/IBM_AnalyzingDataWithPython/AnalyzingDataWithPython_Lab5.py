# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:48:38 2020

@author: Agnieszka
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end2.csv")

print(df.head())
print(df.columns)

# Libraries for plotting
## %% capture
## ! pip install ipywidgets

from ipywidgets import interact, interactive, fixed, interact_manual

# First, use only numeric data

df1 = df._get_numeric_data()
print(df1.head())

# Functions for plotting

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName)
    
    plt.title(Title)
    plt.xlabel("Price(in dollars)")
    plt.ylabel("Proportion of cars")
    
    plt.show()
    plt.close()

#%%

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    # training data, testing data, 
    # lr: linear regression object, 
    # poly_transform: polynomial transformation object
    
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)
    
    plt.plot(xtrain, y_train, "ro", label="Training data")
    plt.plot(xtest, y_test, "go", label="Test data")
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1,1))), label="Predicted function")
    plt.ylim([-10000,60000])
    plt.ylabel("Price")
    plt.legend()
    
#%%

# 1. Training and testing

# Place the target data "price" in a separate dataframe y:

y_data=df1["price"]

# Drop price data in x data

x_data = df1.drop("price", axis=1)

# x_data: features or independent variables
# y_data: dataset target

# Randomly split data into training and data set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# test_size = percantage of the data for testing (here: 10%)
# random_state = number generator used for random sampling

print("Number of test samples: ", x_test.shape[0]) # Number of test samples:  21
print("Number of training samples: ", x_train.shape[0]) # Number of training samples:  180

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)

print("Number of test samples: ", x_test_1.shape[0]) # Number of test samples:  81
print("Number of training samples: ", x_train_1.shape[0]) # Number of training samples:  120

#%%

# import LinearRegression 

from sklearn.linear_model import LinearRegression

# Create Linear Regression object

lr=LinearRegression()

# Split 10:90
# Fit the model using feature "horsepower"

lr.fit(x_train[["horsepower"]], y_train)

# R^2 on the test data:
    
print(lr.score(x_test[["horsepower"]], y_test))  # R^2 = 0.36339478087886834

# R^2 on the train data: 
    
print(lr.score(x_train[["horsepower"]], y_train))  # R^2 = 0.6622424809407366

# Split 40:60

lr.fit(x_train_1[["horsepower"]], y_train_1)

print(lr.score(x_test_1[["horsepower"]], y_test_1))  # R^2 = 0.7141148824069643

# Split 10:90

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
lr.fit(x_train1[['horsepower']], y_train1)
print(lr.score(x_test1[['horsepower']], y_test1))  # R^2 = 0.7340541663929401

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.1, random_state=1)
lr.fit(x_train1[['horsepower']], y_train1)
print(lr.score(x_test1[['horsepower']], y_test1))  # R^2 = 0.36339478087886834

#%%

# Cross validation score

from sklearn.model_selection import cross_val_score

Rcross = cross_val_score(lr, x_data[["horsepower"]], y_data, cv=4)

# input: object (lr), feature (horsepower), target (price), number of folds/ partitions for cross validation (cv)

# Default scoring is R-squared:

print(Rcross) # Each element in the array = average R^2 value in the fold [0.77477095 0.51725019 0.74924821 0.04771764]

# Calculate the average and standard deviation of the estimate

print("The mean of the folds is: ", Rcross.mean(), " and the standard deviation is: ", Rcross.std())
# The mean of the folds is:  0.5222467481979471  and the standard deviation is:  0.29176230741826353

# Negative squared error as a score:

Rcross_neg = -1 * cross_val_score(lr, x_data[["horsepower"]], y_data, cv=4, scoring="neg_mean_squared_error")
print(Rcross_neg) 
# without multiplication with "-1" [-20240865.23167854 -43737944.37864215 -12470270.77569642 -17574447.8797166 ]
# [20240865.23167854 43737944.37864215 12470270.77569642 17574447.8797166 ]

# Calculate the average R^2 using two folds, find the average R^2 for the second fold utilizing the horsepower as a feature

Rcross2 = cross_val_score(lr, x_data[["horsepower"]], y_data, cv=2)
print(Rcross2.mean())  # 0.5174371732192354 

#%%

# Predicting the output using one fold for testing and the other folds for training

from sklearn.model_selection import cross_val_predict 

yhat = cross_val_predict(lr, x_data[["horsepower"]], y_data, cv=4)
print(yhat[0:5]) # [14144.56553323 14144.56553323 20819.140433   12747.56148444  14765.45622158]

#%%

# 2. Overfitting, Underfitting, Model Selection

# a) MLR

lr = LinearRegression()

# Train the model using 4 features
lr.fit(x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_train)

# Prediction using training data

yhat_train = lr.predict(x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
print(yhat_train[0:5])
# [ 7425.12006788 28327.02313301 14210.29915814  4054.61292262  34498.57562361]

# Prediction using test data

yhat_test = lr.predict(x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
print(yhat_test[0:5])
# [11350.96269501  5884.81415149 11206.85674887  6640.86207975  15566.72516819]

# Evaluation of the model using training and test data separately

import matplotlib.pyplot as plt
# %matplotlib inline -> in Jupiter Notebook: plots will be shown in the notebook
import seaborn as sns

# Examine the distribution of the predicted values

# Training data

Title = "Distribution Plot of  Predicted Value Using Training Data vs Training Data Distribution"
DistributionPlot(y_train, yhat_train, "Actual Values(Train)", "Predicted Values (Train)", Title)

# Model seems to be doing well in learning form the training data

# Test data

Title = "Distribution Plot of Predicted Value Using Test Data vs Test Data Distribution"
DistributionPlot(y_test, yhat_test, "Actual Values(Test)", "Predicted Values (Test)", Title)
#%%

# b) Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

# Split the data

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_data, y_data, test_size=0.45, random_state=0) 

# 5th degree polynomial transformation on the feature "horse power"

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train2[["horsepower"]])
x_test_pr = pr.fit_transform(x_test2[["horsepower"]])
print(pr)

# Create and train linear regression model

poly = LinearRegression()
poly.fit(x_train_pr, y_train2)

# See output of the model

yhat2 = poly.predict(x_test_pr)
print(yhat2[0:5])  # [ 6722.92523963  7301.45663996 12214.1679257  18901.59817083  20000.14002037]

# Compare predicted values to actual targets

print("Predicted values: ", yhat2[0:4]) # Predicted values:  [ 6722.92523963  7301.45663996 12214.1679257  18901.59817083]
print("True values: ", y_test2[0:4].values) # True values:  [ 6295 10698 13860 13499]

### * output with .values: True values:  [ 6295 10698 13860 13499]
### * output without .values: True values:  18      6295
###                           170    10698
###                           107    13860
###                           98     13499
###                           Name: price, dtype: int64


# Display the training and testing data and the prediction function using the "PollyPlot" function

PollyPlot(x_train2[["horsepower"]], x_test2[["horsepower"]], y_train2, y_test2, poly, pr)

# the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points.


# Calculate R^2 of the training data

print(poly.score(x_train_pr, y_train2)) # 0.557175738532989

# Calculate R^2 of the test data

print(poly.score(x_test_pr, y_test2)) # -29.624596453924124
# The lower the R^2, the worse the model; negative R^2 is a sign of overfitting

#%%

# Check how the R^2 changes on the test data for different order polynomials:
    
Rsqu_test = [] # create an empty list to store the values

order = [1,2,3,4] # list containing different polynomial orders

for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train2[["horsepower"]]) # transform the training and test data into polynomial
    x_test_pr = pr.fit_transform(x_test2[["horsepower"]])
    
    lr.fit(x_train_pr, y_train2) # fit the regression model
    
    Rsqu_test.append(lr.score(x_test_pr, y_test2)) # Calculate R^2 using the test data and store in the array

plt.plot(order, Rsqu_test)
plt.xlabel("order")
plt.ylabel("R^2")
plt.title("R^2 Using Test Data")
plt.text(3, 0.75, "Maximum R^2")

#%%

# Function to experiment with different polynomial orders and amounts of data

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05)) 
# Interact function (ipywidgets.interact) works only in Jupyter Notebook; two sliders are generated to change the parameters
# and check the diagrams and choose the best order and ratio of test / training data

#%%

# Polynomial transformations with more than one feature. 
# Create a "PolynomialFeatures" object of degree two

pr1 = PolynomialFeatures(degree=2)

# Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. 

x_train_pr1 = pr1.fit_transform(x_train2[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
x_test_pr1 = pr1.fit_transform(x_test2[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])

# Check the dimensions of the new feature

print(x_train_pr1.shape) # (110, 15) -> 15 features

# Create a linear regression model and train the object using the method "fit" using the polynomial features

poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train2)

# poly1= LinearRegression().fit(x_train_pr1, y_train2)

# Use the method "predict" to predict an output on the polynomial features, then use the function "DistributionPlot" 
# to display the distribution of the predicted output vs the test data?

yhat_test1 = poly1.predict(x_test_pr1)
Title="Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data"
DistributionPlot(y_test2, yhat_test1, "Actual Values(Test)", "Predicted Values (Test)", Title)

#The predicted value is higher than actual value for cars where the price $10,000 range, 
# conversely the predicted price is lower than the price cost in the $30,000 to $40,000 range. 
# As such the model is not as accurate in these ranges.

#%%

# 3. Ridge regression

# 2nd degree polynomial transformation 

pr3=PolynomialFeatures(degree=2)
x_train_pr2=pr3.fit_transform(x_train2[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr2=pr3.fit_transform(x_test2[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

from sklearn.linear_model import Ridge

# Create ridge regression object 

RidgeModel = Ridge(alpha=0.1)

# Fit the model

RidgeModel.fit(x_train_pr2, y_train2)

# Obtain a prediction

yhat3 = RidgeModel.predict(x_test_pr2)

# Compare predicted values

print("predicted: ", yhat3[0:4]) # predicted:  [ 6574.0464582   9587.99203464 20825.37570771 19346.56889242]
print("test set: ", y_test2[0:4].values) # test set:  [ 6295 10698 13860 13499]
#%%

# Select the value of Alpha that minimizes the test error

Rsqu_test1 = []
Rsqu_train1 = []
dummy1 = []
Alpha = 10*np.array(range(0,1000))
# print(Alpha) # array -> [0, 10, 20, 30, ..., 9990]

for alpha in Alpha:
    RidgeModel = Ridge(alpha = alpha)
    RidgeModel.fit(x_train_pr2, y_train2)
    Rsqu_test1.append(RidgeModel.score(x_test_pr2, y_test2))
    Rsqu_train1.append(RidgeModel.score(x_train_pr2, y_train2))
    
# Plot the R^2 value for different Alphas

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha, Rsqu_test1, label = "Validation data") # Here: Test data = validation data
plt.plot(Alpha, Rsqu_train1, "r", label = "Training data")
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.legend()
    
# The model is built and tested on the same data. 
# Red line - R^2 of the training data -> 
# As Alpha increases the R^2 decreases. Therefore as Alpha increases the model performs worse on the training data.
# Blue line - R^2 of the validation (test) data ->
# As the value for Alpha increases the R^2 increases and converges at a point 

#%%

# Perform Ridge regression and calculate the R^2 using the polynomial features, use the training data to train the model and test data to test the model. 
# The parameter alpha should be set to 10.
    
RidgeModel1 = Ridge(alpha=10) 
RidgeModel1.fit(x_train_pr2, y_train2)
print(RidgeModel1.score(x_test_pr2, y_test2))    # R^2 = 0.5417231204484027

#%%

# Grid Search

from sklearn.model_selection import GridSearchCV

# Create dictionary of parameter values

parameters = [{"alpha": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]
print(parameters)

# Create a Ridge Regression object

RR = Ridge()
RR

# Create ridge grid search object

Grid = GridSearchCV(RR, parameters, cv=4)

# Fit the model

Grid.fit(x_data[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_data)

# 

BestRR=Grid.best_estimator_

# Test the model on the test data

print(BestRR.score(x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_test))
# 0.6371863176346457 (IBM website: 0.8411649831036152)


