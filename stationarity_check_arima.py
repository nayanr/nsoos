# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:10:51 2019

@author: n0r00te
"""
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import pyplot
df = pd.read_csv("resampled53542044_6881.csv")
#pip install statsmodels



series = pd.Series(df['quantity'].values)


#series.plot()
#pyplot.show()


####AUGMENTED DICKEY FULLER TEST

X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


pip install tensorflow-gpu
# non stationarity
    
import keras
    
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from keras import backend as K
K.tensorflow_backend._get_available_gpus()