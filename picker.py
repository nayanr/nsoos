# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:58:24 2019

@author: n0r00te
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

def date_range(start_date, end_date, increment, period):
    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result

start_date = datetime(2019,5,5)
end_date = start_date + relativedelta(days=30)
date_list = date_range(start_date, end_date, 12, 'hours')

import pandas as pd
data = pd.read_csv("53542044_6881.csv")

data['orderDate'] = pd.to_datetime(data['orderDate'])
data = data.sort_values(by=['orderDate'],ascending=True)
data.head() 



import pytz
utc=pytz.UTC


from datetime import tzinfo, timedelta, datetime

data.shape

#print(date_list)

list_quantity = []
count = 0

data['orderDate'] = pd.to_datetime(data.orderDate)

data = data.reset_index()

for i in range(0, len(date_list)):
    
#    data_window = data[(data['orderDate'] >= date_list[i].replace(tzinfo = utc)) & (data['orderDate'] < date_list[i+1].replace(tzinfo = utc))]
#    data_window = data['orderDate'].between_time(date_list[i].replace(tzinfo = utc), date_list[i+1].replace(tzinfo = utc)) 
    
    if i==len(date_list)-1:
        list_quantity.append(-9999)
        continue
        
    start_date = pd.to_datetime(date_list[i])

    end_date = pd.to_datetime(date_list[i+1])
    
    print(start_date, end_date)
    
    data_window = data.loc[(data['orderDate'] > start_date.replace(tzinfo = utc)) & (data['orderDate'] < end_date.replace(tzinfo = utc))]
    
    if len(data_window) == 0:
        list_quantity.append(-9999)
        continue
    
    latest_entry = data_window.orderDate.idxmax()
                
    list_quantity.append(data.iloc[latest_entry]['inventory_quantity'])
    


len(list_quantity)

df = pd.DataFrame()

df['date'] = date_list
df['quantity'] = list_quantity

df = df[df.quantity>0]

df.to_csv("12hourresampled53542044_6881.csv")
df.head()

df






#import matplotlib.pyplot as plt
#ax = plt.gca()
#
#df.plot(kind='line',x='date',y='quantity',ax=ax)
#plt.show()




