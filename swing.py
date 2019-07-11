# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:32:45 2019

@author: n0r00te
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from numpy import array
from dateutil.relativedelta import relativedelta

import pytz
utc=pytz.UTC
from datetime import tzinfo, timedelta, datetime




#function used for the resampling part
def date_range(start_date, end_date, increment, period):
    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result



#loading master dataset
df = pd.read_csv("restart.csv")


list_1 = df.inventory_offerId.unique()
print(len(list_1))


listmse = []
listrmse = []
listmape = []
listinv = []
listitem = []
listmae = []

count = 0


for item_1 in list_1:
    
# try:
    df1 = df.loc[df['inventory_offerId'] == item_1]
    
    list_2 = df1.inventory_distributorId.unique()
    print(len(list_2))
    
    for item_2 in list_2:
        
      df2 = df1.loc[df1['inventory_distributorId'] == item_2]
        
      
        
      if(df2.shape[0]>2000):
#       if(count<10):
        print(count)
        print(df2.shape)
        data = df2[['orderDate','inventory_quantity']]
        
        
        
        print("resampling part")
        
        
        start_date = datetime(2019,5,5)
        end_date = start_date + relativedelta(days=30)
        date_list = date_range(start_date, end_date,0.5, 'hours')
        
        data['orderDate'] = pd.to_datetime(data['orderDate'])
        data = data.sort_values(by=['orderDate'],ascending=True)
        data.head()
        
        
        if(data.empty):
            continue
        
        list_quantity = []
        count = 0
        for item in date_list:
            count = count + 1
            date_a = item
            last = -10000
            for i, row in data.iterrows():
                date_b =(row["orderDate"])
                date_b = date_b.replace(tzinfo = utc)
                date_a = date_a.replace(tzinfo = utc)
                #print(date_b)
                if((date_a)>(date_b)):
                    last = row["inventory_quantity"]
                    
            print(last,count)        
            list_quantity.append(last)        
        
        
        print("resampling done")
        ###data dataframe  has been updated here
        
        
        data = pd.DataFrame()
        
        
        
        data['date'] = date_list
        data['quantity'] = list_quantity        
        data = data[data.quantity>0]
        
        
#        data.rename(columns={'inventory_quantity':'quantity',
 #                         'orderDate':'orderDate'}, inplace=True)
        
        data.set_index(['date'], inplace=True)
#        data = data[data.quantity>0]
#        data = data[['quantity']]
        
        
#### code for data plot
        
#        sns.set_style("ticks")
#        plt.figure(figsize=(25,5))
#        plt.plot(df['quantity'])
        
        data['quantity(t-0)'] = data.quantity.shift(1)
        data['quantity(t-1)'] = data.quantity.shift(2)
        data['quantity(t-2)'] = data.quantity.shift(3)
        data['quantity(t-3)'] = data.quantity.shift(4)
        
        
        data = data.dropna()
        
        if(data.empty):
            continue
        
        test_set = data[-300:]
        train_set = data[:-300]
        X_test_set = test_set.drop(['quantity'], axis=1)
        y_test_set = test_set['quantity']
        X_train_set = train_set.drop(['quantity'], axis=1)
        y_train_set = train_set['quantity']


        X = array(X_train_set)
        y = array(y_train_set)
        
        
        X = X.reshape((X.shape[0], X.shape[1],1))
        
        
        
        
        print("into training")
        model = Sequential()
        model.add(LSTM(128, activation = 'relu', input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, validation_split=0.10)
        
        
        
        
        x_test = array(X_test_set)
        x_input = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        prediction = model.predict(x_input, verbose=0)
        print(prediction)
        
        
        y_test_set = array(y_test_set)
        y_test_set   
        
        prediction.flatten()
        
        
        actual = np.append(y,y_test_set)
        predicted = np.append(y, prediction)
        
        
        
        
        
        
        
        
        
        def _error(actual: np.ndarray, predicted: np.ndarray):
        #    """ Simple error """
            return actual - predicted
        
        
        def mse(actual: np.ndarray, predicted: np.ndarray):
        #    """ Mean Squared Error """
            return np.mean(np.square(_error(actual, predicted)))
        
        
        def rmse(actual: np.ndarray, predicted: np.ndarray):
        #    """ Root Mean Squared Error """
            return np.sqrt(mse(actual, predicted))
        EPSILON = 1e-10
        def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
        #    """
        #    Percentage error
        #    Note: result is NOT multiplied by 100
        #    """
            return _error(actual, predicted) / (actual + EPSILON)
        
        
        
        def mape(actual: np.ndarray, predicted: np.ndarray):
        #    """
        #    Mean Absolute Percentage Error
        #    Properties:
        #        + Easy to interpret
        #        + Scale independent
        #        - Biased, not symmetric
        #        - Undefined when actual[t] == 0
        #    Note: result is NOT multiplied by 100
        #    """
            return np.mean(np.abs(_percentage_error(actual, predicted)))
        def mean_absolute_percentage_error(y_test_set, prediction): 
            return np.mean(np.abs((y_test_set - prediction) / y_test_set)) * 100
        
#        print(mse(actual,predicted))
#        print(rmse(actual,predicted))
#        print(mape(actual,predicted))
#        
        listinv.append(item_1)
        listitem.append(item_2)
        listmse.append(mse(y_test_set,prediction))
        listrmse.append(rmse(y_test_set,prediction))
        listmape.append(100*mape(y_test_set,prediction))
        listmae.append(mean_absolute_percentage_error(y_test_set, prediction))
        
        #count = count + 1
        ### prediction and plotting
        
        
        
#        prediction.flatten()
#        
#        
#        
#        sns.set_style("ticks")
#        plt.figure(figsize=(25,5))
#        plt.plot(y_test_set, label = "Actual")
#        plt.plot(prediction, label = "Predicted")
#        plt.show()
#        
#        
#        actual = np.append(y,y_test_set)
#        predicted = np.append(y, prediction)
#        
#        sns.set_style("ticks")
#        plt.figure(figsize=(25,5))
#        plt.plot(actual, label = "Actual")
#        plt.plot(predicted, label = "Predicted", alpha=0.7)
#        plt.legend()
#        plt.show()
        
        
        
        #####for a single input prediction
#        x_input = array([time-4, time-3, time-2,time-1])
#        x_input = x_input.reshape((1, n_steps, n_features))
#        yhat = model.predict(x_input, verbose=0)
#        print(yhat)
         
# except:
#     pass
 
result = pd.DataFrame()
result['offerid'] = listitem  
result['distributorid'] = listinv
result['mse'] = listmse 
result['rmse'] = listrmse
result['mape'] = listmape 
result['mae'] = listmae
 

result.to_csv("stats.csv")
    


