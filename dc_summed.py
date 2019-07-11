# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:54:33 2019

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



#df.set_index(['date'], inplace=True)
#df = df[df.quantity>0]
df = df[['total_quantity']]
df.head()

df

sns.set_style("ticks")
plt.figure(figsize=(25,5))
plt.plot(df['total_quantity'])




#### """"'""""INTEGRATING SALES """""""""""""#######


#last = 0
#list = []
#for i, row in df.iterrows():
#    if(i==0):
#        nc = row['quantity']
#        last = nc
#        list.append(nc)
#    else:
#        current = row['quantity']
#        nc = last - current
#        last = current
#        list.append(nc)
#        
#df['diff_from_last'] = list
#
#list = []
#total = 0;
#for i, row in df.iterrows():
#    current = row['diff_from_last']
#    if(i==0):
#        starti = current
#        total = starti
#    else:
#        if(current<0):
#            total = total + current*(-1);
#print(total)   
#
#
#list = []
#for i, row in df.iterrows():
#    current = row['diff_from_last']
#    if(i==0):
#        list.append(total)
#    else:
#        if(current>0):
#            total = total - current;
#            list.append(total)
#        else:
#            list.append(total)
#            
#
#df['salesapx'] = list




df['quantity(t-0)'] = df.total_quantity.shift(1)
df['quantity(t-1)'] = df.total_quantity.shift(2)
df['quantity(t-2)'] = df.total_quantity.shift(3)
df['quantity(t-3)'] = df.total_quantity.shift(4)
#df['quantity(t-4)'] = df.total_quantity.shift(5)
# 
#df['quantity(t-5)'] = df.quantity.shift(6)
#df['quantity(t-6)'] = df.quantity.shift(7)
#df['quantity(t-7)'] = df.quantity.shift(8)
#df['quantity(t-8)'] = df.quantity.shift(9)

#df['quantity(t-9)'] = df.quantity.shift(10)
#df['quantity(t-10)'] = df.quantity.shift(11)
#df['quantity(t-11)'] = df.quantity.shift(12)
#df['quantity(t-12)'] = df.quantity.shift(13)


df = df.dropna()


df
#listn = []
#last = 0
#current = 0
#
#for i, row in df.iterrows():
#    if(last==0):
#        listn.append(-99999)
#        last = row['quantity']
#        
#    else:
#        current = row['quantity']
#        if(current-last>=40):
#            print(current,last)
#            listn.append(1)
#        else:
#            listn.append(0)
#        last = current
#
##df['replenishment'] = listn
#
##df =  df[df['replenishment']>=0]
#
##
#
#
#count = 0
#listr = []
#for i,row in df.iterrows():
#    if(row['replenishment'] == 1):
#        listr.append(count)
#    count = count + 1
#
#
#listr
#count = 0
#listnext = []
#for i,row in df.iterrows():
#    
#    if(row['replenishment'] == 0):
#       for item in listr:
#           if(count<item):
#               listnext.append(item-count)
#               break
#           if(item == listr[len(listr)-1]):
#               listnext.append(999999)
#           
#            
#    else:
#        listnext.append(0)
#        
#            
#    
#    count = count + 1
#    
#
#listnext
#df['d_to_nextr'] = listnext


df

#df['nextr(t-3)'] = df.d_to_nextr.shift(1)
#df['nextr(t-3)'] = df.d_to_nextr.shift(2)
#df['nextr(t-3)'] = df.d_to_nextr.shift(3)
#df['nextr(t-3)'] = df.d_to_nextr.shift(4)

df = df.dropna()

#df = df[df['d_to_nextr']<999999]
test_set = df[-10:]
train_set = df[:-10]
X_test_set = test_set.drop(['total_quantity'], axis=1)
y_test_set = test_set['total_quantity']
X_train_set = train_set.drop(['total_quantity'], axis=1)
y_train_set = train_set['total_quantity']

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
#model = Sequential()

y_test_set = array(y_test_set)
len(y_test_set)

model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape=(X.shape[1], 1),return_sequences = False))
##model.add(Bidirectional(LSTM(50,input_shape=(X.shape[1], 1))))
#model.add(LSTM(50))
model.add(Dense(200))
model.add(Dense(1))

model.summary()
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy','mae'])




model.fit(X, y, epochs = 500)
x_test = array(X_test_set)
x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
prediction = model.predict(x_input, verbose=1)



print(mean_absolute_percentage_error(y_test_set, prediction))

y_test_set



dfsee = pd.DataFrame()
dfsee['actual'] = y_test_set
dfsee['predicted'] = prediction
dfsee.to_csv("dfsee.csv")

X_test_set



import queue



l = queue.Queue(maxsize = 5)

l.put(1051)
l.put(1084)
l.put(1124)
l.put(1170)
listh = []
for i in range(0,10):
    print(i)
    listc = []
    while(l.empty() == False):
        listc.append(l.get())
    new_input = np.asarray(listc)
    new_input = new_input.reshape((1, 4, 1))
    yhat = model.predict(new_input, verbose=0)
    for j in range(1,len(listc)):
        l.put(listc[j])
    l.put(yhat)
    listh.append(yhat)
    i = i + 1

yhat
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
#plt.plot(predicted, label = "Predicted_single_step", alpha=0.7)
plt.plot(predicted_ms,label = "Multistep",color = 'brown')
plt.legend()
plt.show()


####################################################################################

#ROLLING MEAN FOR FILTERING REPLENISHMENT POINTS





