# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:32:58 2020

@author: Agnieszka
"""

import pandas as pd
import numpy as np


df = pd.read_csv("C:/Users/Agnieszka/Downloads/Datasets/auto.csv", header = None)

df.info()
print(df.tail(10))

# List with column names (header)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# Adding a header

df.columns = headers

print(df.head(10))

print(df.columns)

# Replacing the "?" symbol with NaN
# so the dropna() can remove the missing values

df1 = df.replace('?', np.NaN)

# Dropping missing values along the column "price"
# axis=0 -> drops the entire row, axis=1 -> drops the column

df = df1.dropna(subset=["price"], axis=0)

# 4 rows were dropped

df.info()

# Saving dataset to csv; index = False -> without column with names of statistics

df.to_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header.csv", index = False)

# Checking data type

print(df.dtypes) 

# Statistical summary of each numerical column(count, mean, standard deviation, etc.)

print(df.describe())

#stat = df.describe()
#stat.to_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_stat.xlsx")

# Statistical summary of all columns

df.describe(include = "all")

# stat = df.describe(include = "all")
# stat.to_excel("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_stat2.xlsx")

# Choosing certain column for statistics


df[['length', 'compression-ratio']].describe()

print(df[['length', 'compression-ratio']].describe())