# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:21:49 2020

@author: srine
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgboost import plot_tree
from sklearn import metrics
from sklearn.feature_selection import RFECV
from xgboost import plot_importance

#from pandas import read_csv, set_option
from sklearn.model_selection import train_test_split

stock_prices = "C:/Users/srine/OneDrive/Desktop/FYP/JPM-XGBoost.csv"
stock_data = pd.read_csv(stock_prices, parse_dates=[0])
stock_data.head()

# Check the shape and datatypes of the stock prices dataframe
print(stock_data.shape)
print(stock_data.dtypes)

merged_dataframe = stock_data[['Date','Open', 'High', 'Adj Close', 'Volume','Low' ,'Close','EPS','PE Ratio']]

pd.options.mode.chained_assignment = None

merged_dataframe['Year'] = pd.DatetimeIndex(merged_dataframe['Date']).year
merged_dataframe['Day'] = pd.DatetimeIndex(merged_dataframe['Date']).month
merged_dataframe['Month'] = pd.DatetimeIndex(merged_dataframe['Date']).day

# Check the shape and top 5 rows of the merged dataframe
print(merged_dataframe.iloc[:5])
print(merged_dataframe.shape)
print(merged_dataframe.dtypes)

# Check the statistics of the columns of the merged dataframe and check for outliers
print(merged_dataframe.describe())

#Data Preprocessing
merged_dataframe['Adj Factor'] = merged_dataframe['Adj Close'] / merged_dataframe['Close']
merged_dataframe['Open'] = merged_dataframe['Open'] / merged_dataframe['Adj Factor']
merged_dataframe['High'] = merged_dataframe['High'] / merged_dataframe['Adj Factor']
merged_dataframe['Low'] = merged_dataframe['Low'] / merged_dataframe['Adj Factor']
merged_dataframe['Volume'] = merged_dataframe['Volume'] / merged_dataframe['Adj Factor']
merged_dataframe['Adj Close shift'] = merged_dataframe['Adj Close'].shift(-1)
merged_dataframe['Open shift'] = merged_dataframe['Open'].shift(-1)
merged_dataframe['high_diff'] = merged_dataframe['High'] - merged_dataframe['Adj Close shift']
merged_dataframe['low_diff'] = merged_dataframe['Low'] - merged_dataframe['Adj Close shift']
merged_dataframe['close_diff'] = merged_dataframe['Adj Close'] - merged_dataframe['Adj Close shift']
merged_dataframe['open_diff'] = merged_dataframe['Open shift'] - merged_dataframe['Adj Close shift']

# Separate the dataframe for input(X) and output variables(y)
X = merged_dataframe[['Open', 'High', 'Adj Close', 'Volume','Low','Day','Year','Month','Adj Factor','Adj Close shift','Open shift','high_diff','low_diff','close_diff','open_diff','EPS','PE Ratio']]
y = merged_dataframe.loc[:,'Close']

#print(merged_dataframe['Date'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train=np.array(X_train)
y_train=np.array(y_train)
y_test=np.array(y_test)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)
X_test = X_test.astype(float)

np.nan_to_num(X_train)
np.nan_to_num(y_train)
np.nan_to_num(X_test)
np.nan_to_num(y_test)

model = XGBRegressor(booster='gblinear', reg_lambda=0, objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(np.nan_to_num(X_train), np.nan_to_num(y_train))
preds = model.predict(np.nan_to_num(X_test))

print('Mean Absolute Error    :', metrics.mean_absolute_error(np.nan_to_num(y_test), np.nan_to_num(preds)))  
print('Mean Squared Error     :', metrics.mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(preds)))
rmse = np.sqrt(mean_squared_error(np.nan_to_num(y_test), np.nan_to_num(preds)))
print("Root Mean Square Error : %f" % (rmse))

accuracy = 100 - np.mean(np.abs((y_test - preds) / y_test)) * 100
print("ACCURACY OF XGBoost Model: " + str(accuracy))

print(model.feature_importances_)

print(plot_importance(model))

fig, ax = plt.subplots(figsize=(75, 75)) 
plot_tree(model,ax=ax)
plt.savefig('C:/Users/srine/OneDrive/Desktop/FYP/jpm_xgb.png')
plt.show()

imp = model.feature_importances_
imp_list = imp.tolist()
list(np.float_(imp_list))
print(imp_list)
label = ['Open', 'High', 'Adj Close', 'Volume','Low','Day','Year','Month','Adj Factor','Adj Close shift','Open shift','high_diff','low_diff','close_diff','open_diff','EPS','PE Ratio']
   
res = sorted(range(len(imp_list)), key = lambda sub: imp_list[sub])[-5:] 
# printing result 
print("Indices list of max N elements is : " + str(res))

i=0
selected=[]
for val in res:
        selected.append(label[val])

print(selected)

# plot importance features
index = ['Open', 'High', 'Adj Close', 'Volume','Low','Day','Year','Month','Adj Factor','Adj Close shift','Open shift','high_diff','low_diff','close_diff','open_diff','EPS','PE Ratio']
df = pd.DataFrame({
        'Attribute':index,
        'Importance of Feature':model.feature_importances_})
df.plot(kind='bar',x='Attribute',y='Importance of Feature')

# Plot Prediction vs Actual
plt.plot(y_test,color='blue',label ="Actual Value")
plt.plot(preds,color='red', label="Predicted Value")
plt.xlabel('Time Series')
plt.ylabel('Stock Value')
plt.legend(loc="bottom right")
plt.show()

af = merged_dataframe['Adj Factor'].mean()
acs = merged_dataframe['Adj Close shift'].mean()
os = merged_dataframe['Open shift'].mean()

# Predict Results
print("Enter the following Values for Prediction:\n")
Open = float(input("Open: "))
High = float(input("High: "))
AdjClose = float(input("Adjusted Close: "))
Volume = float(input("Volume: "))
Low = float(input("Low: "))
Day = float(input("Day: "))
Year = float(input("Year: "))
Month = float(input("Month: "))

Open = Open/af
High = High/af
Low = Low/af
Volume = Volume/af

AdjFactor = af
AdjCloseshift = acs
Openshift = os
high_diff = High - AdjCloseshift
low_diff = Low - AdjCloseshift
close_diff = AdjClose - AdjCloseshift
open_diff =  Openshift - AdjCloseshift
EPS = float(input("EPS: "))
PERatio = float(input("P/E Ratio: "))

print(model.predict([Open, High, AdjClose, Volume,Low,Day,Year,Month,AdjFactor,AdjCloseshift,Openshift,high_diff,low_diff,close_diff,open_diff,EPS,PERatio]))