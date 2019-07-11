# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:41:05 2019

@author: n0r00te
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pytz
utc=pytz.UTC
from datetime import tzinfo, timedelta, datetime


data = pd.read_csv("restart.csv")


data = data[['inventory_quantity','inventory_offerId','inventory_distributorId','orderDate','inventory_distributorType']]

data['orderDate'] = pd.to_datetime(data['orderDate'])
data = data.sort_values(by=['orderDate'],ascending=True)

data.set_index(['orderDate'], inplace=True)


data.dtypes

X = data.iloc[:, 1:5].values
X
y = data.iloc[:, 0].values
y

#three categorical values inventory_offerId,inventory_distributorId,and inventory_distributorType
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])



#the model can confuse between catergorical distributor types and inventory types
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()


#proceeding without one hot encoding


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train

#no features to be scaled

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from numpy import array


X = array(X_train)
y = array(y_train)


X = X.reshape((X.shape[0], X.shape[1],1))


model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, validation_split=0.10)

x_test = array(X_test)
x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_input, verbose=0)
print(prediction)


y_test_set = array(y_test)
y_test_set


prediction.flatten()



sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(y_test_set, label = "Actual")
plt.plot(prediction, label = "Predicted")
plt.show()


actual = np.append(y,y_test_set)
predicted = np.append(y, prediction)

sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(actual, label = "Actual")
plt.plot(predicted, label = "Predicted", alpha=0.7)
plt.legend()
plt.show()

