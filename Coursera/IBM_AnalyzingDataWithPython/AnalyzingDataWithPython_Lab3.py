# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:59:11 2020

@author: Agnieszka
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Agnieszka/Downloads/Datasets/auto_with_header_cleaned_end2.csv")

#%%
df.info()
print(df.head())

# Checking correlation between continuous variables (int64 / float64)
print(df.corr()) # output - correlation coefficients

print(df[["bore","stroke","compression-ratio","horsepower"]].corr())
#%%

#%%
# Scatterplots

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

print(df[["engine-size", "price"]].corr())
#%%

sns.regplot(x="highway-mpg", y="price", data=df)
print(df[["highway-mpg", "price"]].corr())

#%%

## Positive/negative relationship between engine-size/ highway-mpg and the price
## It means, they can be used to predict price

sns.regplot(x="peak-rpm", y="price", data=df)
print(df[["peak-rpm","price"]].corr())

# peak-rpm is not a good predictor - regression line close to horizontal, correlation very low (-0,102)

#%%

print(df[["stroke","price"]].corr())
sns.regplot(x="stroke", y="price", data=df)

# no linear relationship

#%%

# Checking correlation between categorical variables (object/int64)

# Boxplots

sns.boxplot(x="body-style", y="price", data=df)
# The distributions of price between the different body-style categories have a
# significant overlap, and so body-style would not be a good predictor of price. 

sns.boxplot(x="engine-location", y="price", data=df)

# The distribution of price between these two engine-location categories, 
# front and rear, are distinct enough to take engine-location as a potential 
# good predictor of price.

sns.boxplot(x="drive-wheels", y="price", data=df) 

# The distribution of price between the different drive-wheels categories differs; 
# as such drive-wheels could potentially be a predictor of price. 

#%%

print(df.describe()) # without variables of type object
print(df.describe(include=["object"])) # only variables of type object

#%%

# Value counts

print(df["drive-wheels"].value_counts()) 

# value_counts method works only on Pandas series not dataframes; that's the reason why we use only one bracket df[".."]

# converting series to dataframe

print(df["drive-wheels"].value_counts().to_frame())

#%%

# generating new dataframe with counts of unique values of drive-wheels variable
drive_wheel_counts = (df["drive-wheels"].value_counts().to_frame())

drive_wheel_counts.rename(columns={"drive-wheels":"value_counts"}, inplace=True)
print(drive_wheel_counts)


drive_wheel_counts.index.name = "drive-wheels"
print(drive_wheel_counts)
#%%
# generating new dataframe with counts of unique values of engine location variable
engine_loc_counts = (df["engine-location"].value_counts().to_frame())
engine_loc_counts.rename(columns={"engine-location":"value_counts"}, inplace=True)
engine_loc_counts.index.name = "engine-location"
print(engine_loc_counts)

#%%

## Group by method

print(df["drive-wheels"].unique())

df_group_one = df[["drive-wheels", "body-style", "price"]]

# calculating average price for each of the different categories of data

df_group_one = df_group_one.groupby(["drive-wheels"], as_index=False).mean()
print(df_group_one)

# output: average prices (one column) for different dirve-wheels categories (one column) - 4wd, fwd, rwd

df_group_two = df[["body-style", "price"]]
grouped = df_group_two.groupby(["body-style"], as_index=False).mean()
print(grouped)

#%%

# grouping with multiple variables variable

df_gptest = df[["drive-wheels", "body-style", "price"]]

grouped_test1 = df_gptest.groupby(["drive-wheels", "body-style"], as_index=False).mean()
#print(grouped_test1)

# Converting a dataframe to a pivot table for better visualisation

grouped_pivot = grouped_test1.pivot(index="drive-wheels", columns="body-style")
print(grouped_pivot)

# Filling missing values (NaN) with 0 

grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

#%%

# Heat map

#
#plt.pcolor(grouped_pivot, cmap="RdBu")
#plt.colorbar()
#plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap="RdBu")

# label names

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor = False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor = False)

# insert labels

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#%%

from scipy import stats

# Calculating pearson correlation coeff. and P-value

pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Coeff = 0.585; p-value < 0.001 => the correlation is statistically significant (p-value), 
# although the linear relationship isn't extremely strong (coeff)

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Coeff = 0.81; p-value < 0.001 => the correlation is statistically significant (p-value), 
# and the linear relationship is quite strong (coeff)

pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Coeff = 0.87; p-value < 0.001 => the correlation is statistically significant (p-value), 
# and the linear relationship is very strong (coeff)

pearson_coef, p_value = stats.pearsonr(df["bore"], df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Coeff = 0.54; p-value < 0.001 => the correlation is statistically significant (p-value), 
# and the linear relationship is moderate (coeff)

#%%

#ANOVA

# Checking if the type of drive-wheels impacts the price

df_gptest = df[["drive-wheels", "body-style", "price"]]
grouped_test2 = df_gptest[["drive-wheels", "price"]].groupby(["drive-wheels"])
print(grouped_test2.head())

print(df_gptest)

print(grouped_test2.get_group("4wd")["price"]) # shows all rows with value "4wd" and its price

# ANOVA, calculating F-test score and P-value

f_score, p_val = stats.f_oneway(grouped_test2.get_group("fwd")["price"], grouped_test2.get_group("rwd")["price"], grouped_test2.get_group("4wd")["price"])
print( "ANOVA results: F=", f_score, ", P =", p_val) 
# F-score = 67.95 ; p-value < 0.001

f_score, p_val = stats.f_oneway(grouped_test2.get_group("fwd")["price"], grouped_test2.get_group("rwd")["price"])
print( "ANOVA results: F=", f_score, ", P =", p_val) 
# F-score = 130.55 ; p-value < 0.001

f_score, p_val = stats.f_oneway(grouped_test2.get_group("rwd")["price"], grouped_test2.get_group("4wd")["price"])
print( "ANOVA results: F=", f_score, ", P =", p_val) 
# F-score = 8.58 ; p-value = 0.004 (> 0.001)


f_score, p_val = stats.f_oneway(grouped_test2.get_group("fwd")["price"], grouped_test2.get_group("4wd")["price"])
print( "ANOVA results: F=", f_score, ", P =", p_val) 
# F-score = 0.67 ; p-value = 0.416 (> 0.001)

### General conlclusion: important variables: length, width, curb-weigth, engine-size, horsepower, city-mpg, 
### highway-mpg, wheel-base, bore, drive-wheels

