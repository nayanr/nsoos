# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:13:57 2019

@author: n0r00te
"""


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.tsa.arima_model import ARIMA


data = pd.read_csv("53542044.csv")



data.set_index(['date'], inplace=True)
#data = data[data.quantity>0]
data.head()

df = data[['total_quantity']]
df.head()
df.plot()
plt.ylabel('InventoryQuantity')
plt.xlabel('Date')
plt.show()

data[["total_quantity"]] = data[["total_quantity"]].apply(pd.to_numeric)
data = data.astype({"total_quantity": float})
data = data[['total_quantity']]

data.dtypes


q = d = range(0, 10)
p = range(0, 10)

pdq = list(itertools.product(p, d, q))

#pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
#
#print('parameter combinations for ARIMA...')
#print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[4]))


data
pdq


test_data = data[-10:]
train_data = data[:-10]


train_data
test_data




import warnings
#import itertools
#import pandas as pd
#import numpy as np
#import statsmodels.api as sm
#p = d = q = range(0, 10)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 0) for x in list(itertools.product(p, d, q))]
warnings.filterwarnings("ignore") # specify to ignore warning messages


AIC= []
ARIMAX_model = []

for param in pdq:
    
    try:
            mod = ARIMA(train_data.total_quantity, order=param,)
            results = mod.fit()
            AIC.append(results.aic)

            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            ARIMAX_model.append([param])
            
    except:
            pass




print('The smallest AIC is {} for model ARIMAX{}'.format(min(AIC), ARIMAX_model[AIC.index(min(AIC))][0]))
mod = ARIMA(train_data.total_quantity,order=ARIMAX_model[AIC.index(min(AIC))][0])
results = mod.fit()
print(results.summary())
results.aic


results
pred0 = results.predict(start='2019-05-26', dynamic=False)
pred0
#
#
#
#
#
#pred2 = results.forecast()
#pred2_ci = pred2.conf_int()
#pred2
#print(pred0.predicted_mean['2019-06-05':'2019-06-08'])





history = [x for x in train_data]
predictions = list()

len(test_data)
for t in range(len(test_data)):
        model = ARIMA(history, order=(0,2,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        


     
     
          
	
     
     

    
    

    
    
error = mean_squared_error(test_data, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()














