# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:48:24 2020

@author: Agnieszka
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end2.csv")

print(df.head())
print(df.columns)

# Linear regression

# y_hat = a +bX

from sklearn.linear_model import LinearRegression

#%% 

# Create a linear regression object

lm = LinearRegression()
lm

# X predictor variable, Y response variable

X = df[["highway-mpg"]]
Y = df["price"]

# Fit the linear model using highway-mpg

lm.fit(X,Y)

# Output a prediction

Y_hat = lm.predict(X)
print(Y_hat[0:5])  
# Output for  Y_hat [0:5] -> [16236.50464347 16236.50464347 17058.23802179 13771.3045085 20345.17153508]

# Intercept a

print(lm.intercept_) # 38423.305858157386

# Slope b

print(lm.coef_) # [-821.73337832] <- array


# Linear model: price = 38423.31 - 821.73 * highway-mpg

#%%

lm1 = LinearRegression()
print(lm1)


lm1.fit(df[["engine-size"]], df[["price"]])

# or X1 = df[["engine-size"]]
# Y1 = df["price"] 
# lm.fit(X1, Y1)

print(lm1.coef_) # [[166.86001569]]
print(lm1.intercept_) # [-7963.33890628]

# price = -7963.34 + 166.86*engine-size

#%%

# Multiple linear regression 
# 𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4

Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]

# fit the linear model

lm.fit(Z, df["price"])

print(lm.intercept_) # -15831.930960299469
print(lm.coef_) # [53.66247317  4.70938694 81.44600167 36.55016267]

# price = -15831.93 + 53.66*horsepower + 4.71*curb-weight + 81.44*engine-size + 36.55*highway-mpg

lm2 = LinearRegression()
Z2 = df[["normalized-losses", "highway-mpg"]]
Y = df["price"]
lm2.fit(Z2, Y)

# or lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

print(lm2.coef_) # [   1.49789586 -820.45434016]
print(lm2.intercept_) # 38201.31327245728

# price = 38201.31 + 1.50*normalized-losses - 820.45*highway-mpg

#%%

# Model evaluation using Visualization

# Regression plot

width = 12
height = 10

plt.figure(figsize=(width, height))
#sns.regplot(x="highway-mpg", y="price", data=df)
#sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

#print(df[["peak-rpm", "price"]].corr()) # -0.10
# print(df[["highway-mpg", "price"]].corr()) # -0.70 <- stronger correlation with price

print(df[["highway-mpg", "peak-rpm", "price"]].corr())
 
#%%

# Residual Plot

width = 12
height = 10

plt.figure(figsize=(width, height))
sns.residplot(df["highway-mpg"], df["price"])
plt.show()

#%%

# Visualization of Multiple Linear Regression
# Distribution plots

# Make a prediction

Y_hat1 = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df["price"], hist = False, color="r", label= "Actual value")
sns.distplot(Y_hat1, hist=False, color="b", label = "Fitted Values", ax=ax1)

plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")

plt.show()
plt.close()

# The fitted values are reasonably close to the actual values, since the two distributions 
# overlap a bit. However, there is definitely some room for improvement.

#%% 

# Polynomial regression (onedimensional)

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df["highway-mpg"]
y = df["price"]

# Polynomial of the 3rd order

f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p) # -1.557 x^3 + 204.8 x^2 - 8965 x + 1.379e+05

PlotPolly(p, x, y, "highway-mpg")
print(np.polyfit(x, y, 3)) # [-1.55663829e+00  2.04754306e+02 -8.96543312e+03  1.37923594e+05]

#%%

# Polynomial of the 11th order

f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1) # -1.243e-08 x^11  + 4.722e-06 x^10 - 0.0008028 x^9 + 0.08056 x^8 - 5.297 x^7
            # + 239.5 x^6 - 7588 x^5 + 1.684e+05 x^4 - 2.565e+06 x^3 + 2.551e+07 x^2 - 1.491e+08 x + 3.879e+08

PlotPolly(p1, x, y, "highway-mpg")

#%%

# Polynomial regression multidimensional

from sklearn.preprocessing import PolynomialFeatures

# PolynomialFeatures object of degree 2

pr = PolynomialFeatures(degree=2)
print(pr)

Z_pr = pr.fit_transform(Z)
print(Z.shape) # original data (201, 4) -> 201 samples, 4 features

print(Z_pr.shape) # after transformation (201,15) -> 201 samples, 15 features

#%%

# Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a list of tuples

Input = [("scale", StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("model", LinearRegression())]
pipe = Pipeline(Input)
print(pipe)

# Train the pipeline object - normalize data, perform a transform and fit the model simultaneously (with pipeline)
pipe.fit(Z,y)

# Predicting - normalize data, perform a transform and produce a prediction simultaneously (with pipeline)
yhat=pipe.predict(Z)
print(yhat[0:4]) # [13103.67557905 13103.67557905 18229.84126783 10394.17656982]

#%% 
# Create a pipeline that Standardizes the data, then perform prediction using a linear regression model using the features Z and targets y

Input = [("scale", StandardScaler()), ("model", LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z,y)
yhat2 = pipe.predict(Z)
print(yhat2[0:10]) # [13698.95609311 13698.95609311 19056.78572196 10621.59764327, 15519.32197778 13867.78444008 15454.84783873 15972.88040209, 17614.41 85158 10723.08344825]

#%%

# Quantitative measures for In-Sample Evaluation

# 1. Simple linear Regression

# Determinig accuracy of a model:
# R-square R^2

lm.fit(X,Y)
print("The R-square is: ", lm.score(X,Y))
# The R-square is:  0.4965911884339175 -> ca. 49.659% of the variation of the price is explained by this simple linear model "Horsepower_fit"

# MSE (Mean Squared Error) - average of the squares of errors that is the diff. between  actual value (y) and the estimated value(y_hat)

Yhat3 = lm.predict(X)
print("The output of the firs four predicted values is: ", Yhat3[0:4])
# The output of the firs four predicted values is:  [16236.50464347 16236.50464347 17058.23802179 13771.3045085 ]

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df["price"], Yhat3)
print("The MSE of price and predicted value is: ", mse)

# The MSE of price and predicted value is:  31635042.944639895

# 2. Multiple linear regression

# fit the model
lm.fit(Z, df["price"])

# R-square

print("The R-square is: ", lm.score(Z, df["price"]))
# The R-square is:  0.8094390422815301
# 80.896 % of the variation of price is explained by this multiple linear regression "multi_fit"

# MSE

# Prediction:
    
Y_predict_multifit = lm.predict(Z)
print("The MSE of price and predicted value using multifit is: ", mean_squared_error(df["price"], Y_predict_multifit))
# The MSE of price and predicted value using multifit is:  11975165.993303549  

# 3. Polynomial fit

# R-square R^2

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print("The R-squared value is: ", r_squared)
# The R-squared value is:  0.674194666390652

# MSE

print(mean_squared_error(df["price"], p(x)))
# 20474146.426361218

#%%
 
# Prediction and Decision making 

# method fit -> training of the model

# now method predict to produce a prediction

import matplotlib.pyplot as plt
import numpy as np

# Create a new input

new_input = np.arange(1,100,1).reshape(-1,1)  # reshape(-1,1) -> -1 = unknown row, 1 = 1 column

# Fit the model

lm.fit(X,Y)
lm

# Produce a prediction

yhat5 = lm.predict(new_input)
print(yhat5[0:5]) # array [37601.57247984 36779.83910151 35958.10572319 35136.37234487 34314.63896655]

plt.plot(new_input, yhat5)
plt.show()

# R^2: The model with the higher R-squared value is a better fit for the data
# MSE: The model with the smallest MSE value is a better fit for the data

# Here: the best model - Multiple Linear Regression (MLR) to predict price
# This makes sense, since there are 27 variables in total and more than one of those variables are potential
# predictor of the final car price





