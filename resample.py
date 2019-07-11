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
date_list = date_range(start_date, end_date,8, 'hours')

import pandas as pd
data = pd.read_csv("53542044_6881.csv")

data['orderDate'] = pd.to_datetime(data['orderDate'])
data = data.sort_values(by=['orderDate'],ascending=True)
data.head()



import pytz
utc=pytz.UTC


from datetime import tzinfo, timedelta, datetime

data.shape

print(date_list)

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
    
    


len(list_quantity)

df = pd.DataFrame()

df['date'] = date_list
df['quantity'] = list_quantity
df = df[df.quantity>0]





df.to_csv("optz53542044_6881.csv")
df.head()
df = df[df.quantity>0]
df









import matplotlib.pyplot as plt
ax = plt.gca()

df.plot(kind='line',x='date',y='quantity',ax=ax)
plt.show()










