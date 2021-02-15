# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:17:55 2020

@author: Agnieszka
"""

import pandas as pd

import matplotlib.pylab as plt

df = pd.read_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header.csv")

df.info()

import numpy as np

# Replacing all "?" with NaN

df.replace("?", np.nan, inplace = True)

# Detecting missing values

missing_data = df.isnull()
print(missing_data.head(10)) # True = missing value, False = not missing value

# Counting missing values

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
# Dealing with missing values:
    # Replacing with the average (mean)

    # Normalized losses
    
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#print(df["normalized-losses"].head(20))

    # Bore
    
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)

df["bore"].replace(np.nan, avg_bore, inplace = True)

    # Stroke
    
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)

df["stroke"].replace(np.nan, avg_stroke, inplace = True)

    # Horsepower
  
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)

df["horsepower"].replace(np.nan, avg_horsepower, inplace = True)

    # Peak-rpm
    
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm:", avg_peak_rpm)

df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace = True)

    # Number of doors
    # Counts for each unique value
    
print(df["num-of-doors"].value_counts())

        # Output: four 113 / two 86
    
    # Or calculate most common type

print(df["num-of-doors"].value_counts().idxmax())

    # Replacing with the common value (here: "four")
    
df["num-of-doors"].replace(np.nan, "four", inplace = True)

    # Price - droping all rows that do not have price
    
df.dropna(subset = ["price"], axis=0, inplace = True)
df.reset_index(drop = True, inplace=True)
df.to_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned.xlsx")
    
print(df.dtypes)

# Converting data types
    
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

#  Standardization = transforming data into a common format which allows a meaningful comparison

    # transforming mpg to L/100km (L/100km = 235 / mpg)

df["city-L/100km"] = 235/df["city-mpg"]

print(df.head())

df["highway-L/100km"] = 235/df["highway-mpg"]

print(df.head())
print(df.columns)

    # Rename the column (didn't work!)
    
df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)
print(df.columns)
print(df.head)

#df.to_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned2.xlsx")

# Normalization - variables between 0 -1

    # Simple Feature Scaling (x_old/x_max)
    
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()
    
df["height"] = df["height"]/df["height"].max()

print(df[["length", "width", "height"]].head())

# Binning data - transforming continuous numerical variables into discrete categorical "bin"

df["horsepower"] = df["horsepower"].astype(int, copy=True)

print(df.info())

#%%
# %matplotlib inline (??)
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df["horsepower"])

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
#%%


import matplotlib as plt
from matplotlib import pyplot

bins  = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

group_names = ["Low", "Medium", "High"]

df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels = group_names, include_lowest=True)
print(df[["horsepower", "horsepower-binned"]].head(20))

print(df["horsepower-binned"].value_counts())

# Plotting the binned data

pyplot.bar(group_names, df["horsepower-binned"].value_counts())
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

%matplotlib inline

import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df["horsepower"], bins = 3)
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#%%
# Indicator variable (dummy variable)

    # Converting fuel-type into dummy variable
    
print(df.columns)

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

dummy_variable_1.rename(columns={"gas":"fuel-type-gas", "diesel" :"fuel-type-diesel"}, inplace=True)

print(dummy_variable_1.head())

# merge data frame "df" with "dummy_variable_1"

df = pd.concat([df, dummy_variable_1], axis=1)

df.drop("fuel-type", axis=1, inplace=True)
#%%

print(df.head())

#%%


dummy_variable_2 = pd.get_dummies(df["aspiration"])
print(dummy_variable_2.head())

dummy_variable_2.rename(columns={"std":"aspiration-std", "turbo" :"aspiration-turbo"}, inplace=True)

print(dummy_variable_2.head())

df = pd.concat([df, dummy_variable_2], axis=1)

df.drop("aspiration", axis=1, inplace=True)
#%%

df.info()

print(df.head(20))

#df.to_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end.xlsx")

df2 = pd.read_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end.xlsx")

df2.to_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end2.csv")


