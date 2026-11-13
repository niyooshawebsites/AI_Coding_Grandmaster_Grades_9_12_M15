import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read the data from the csv file
missing_values = ['N/A', 'n/a','na', np.nan]
df = pd.read_csv('country_vaccinations.csv', na_values=missing_values)

# # display the first 10 records
print(df.head(10))

# # checking if there are any missing cells across different columns > returns true or false
print(df.isnull().any())
print("-----------------------------------------------------------------------------")
print(df.isnull().sum())

# # the total number of missing cells across the entire DataFrame
print(df.isnull().sum().sum())

# # visualize the missing values with heatmap(optimize by using a subset of data)
subset = df.iloc[:100, :] # taking the first 100 rows for better performance
plt.figure(figsize=(12, 8))
sns.heatmap(subset.isnull(), cbar=True, cmap='viridis')
plt.show()

# Drop rows where all values are NaN
print(df.info())
df = df.dropna(how="all")
print(df.info())

# # fill missing values using backend fill method
df = df.bfill()
print(df.info())

# Interpolate missing values
df.interpolate()
print(df.info())

# Drop all rows with any NaN values
df_dropped = df.dropna()
print(df_dropped)



