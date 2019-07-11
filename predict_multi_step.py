# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:46:24 2019

@author: n0r00te
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:38:41 2019

@author: n0r00te
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("53542044.csv")



df.set_index(['date'], inplace=True)
df = df[df.quantity>0]
df = df[['quantity']]
df.head()

df

sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(df['quantity'])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()



df['quantity(t-0)'] = df.quantity.shift(1)
df['quantity(t-1)'] = df.quantity.shift(2)
df['quantity(t-2)'] = df.quantity.shift(3)
df['quantity(t-3)'] = df.quantity.shift(4)
df['quantity(t-4)'] = df.quantity.shift(5)
 
#df['quantity(t-5)'] = df.quantity.shift(6)
#df['quantity(t-6)'] = df.quantity.shift(7)
#df['quantity(t-7)'] = df.quantity.shift(8)
#df['quantity(t-8)'] = df.quantity.shift(9)
#
#df['quantity(t-9)'] = df.quantity.shift(10)
#df['quantity(t-10)'] = df.quantity.shift(11)
#df['quantity(t-11)'] = df.quantity.shift(12)
#df['quantity(t-12)'] = df.quantity.shift(13)
#df['quantity(t-12)'] = df.quantity.shift(13)




#df[['quantity','quantity(t-0)']] = min_max_scaler.transform(df[['A','B']])

df = df.dropna()




test_set = df[-10:]
train_set = df[:-10]
X_test_set = test_set.drop(['quantity'], axis=1)
y_test_set = test_set['quantity']
X_train_set = train_set.drop(['quantity'], axis=1)
y_train_set = train_set['quantity']

#df = df.drop(['replenishment'],axis = 1)
import warnings
warnings.filterwarnings('ignore')
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

from keras.layers import Bidirectional
def mean_absolute_percentage_error(y_test_set, prediction): 
    return np.mean(np.abs((y_test_set - prediction) / y_test_set)) * 100




X = array(X_train_set)
y = array(y_train_set)


X = X.reshape((X.shape[0], X.shape[1],1))

y_test_set = array(y_test_set)
len(y_test_set)

 

model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape=(X.shape[1], 1),return_sequences = False))
#model.add(Dropout(0.2))
#model.add(LSTM(50,activation = 'relu',return_sequences = False))
model.add(Dense(1,return_sequences = True))
model.add(Dense(10))
#model.add(Dropout(0.02))
model.summary()
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy','mae'])




model.fit(X, y, epochs = 100)
x_test = array(X_test_set)
x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_input, verbose=1)


model.evaluate(x=x_input, y=y_test_set, batch_size=10, verbose=1, sample_weight=None, steps=None)


print(mean_absolute_percentage_error(y_test_set, prediction))

y_test_set

dfsee = pd.DataFrame()
dfsee['actual'] = y_test_set
dfsee['predicted'] = prediction
dfsee.to_csv("dfsee.csv")

X


import queue



l = queue.Queue(maxsize = 5)

l.put(79)
l.put(79)
l.put(79)
l.put(81)
l.put(81)
listh = []
for i in range(0,20):
    print(i)
    listc = []
    while(l.empty() == False):
        listc.append(l.get())
    new_input = np.asarray(listc)
    new_input = new_input.reshape((1, 5, 1))
    yhat = model.predict(new_input, verbose=0)
    for j in range(1,len(listc)):
        l.put(listc[j])
    l.put(yhat)
    listh.append(yhat)
    i = i + 1

len(listh)


dft = pd.DataFrame()
dft['multistep'] = listh
dft['single-step'] = prediction
dft['actual'] = y_test_set


actual = np.append(y,y_test_set)
predicted = np.append(y, prediction)
predicted_ms = np.append(y,listh)
sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(actual, label = "Actual")
plt.plot(predicted, label = "Predicted_single_step", alpha=0.7)
plt.plot(predicted_ms,label = "Multistep")
plt.legend()
plt.show()
dft.to_csv("dft.csv")

