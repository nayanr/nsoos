# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:33:00 2019

@author: n0r00te
"""
import pandas as pd
data = pd.read_csv("53665163_7397.csv")

data['orderDate'] = pd.to_datetime(data['orderDate'])
data = data.sort_values(by=['orderDate'],ascending=True)
data.head()

data = data[["orderDate","inventory_quantity"]]
 
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
date_list = date_range(start_date, end_date,0.5, 'hours')

import pytz
utc=pytz.UTC

listdates = []
listquantity = []
from datetime import tzinfo, timedelta, datetime

for item in date_list:
    item = item.replace(tzinfo = utc)
    index = abs(item - data['orderDate']).idxmin()
    print(index)
    listdates.append(item)
    listquantity.append(data.iloc[index]['inventory_quantity'])
            
df = pd.DataFrame()
df['date'] = listdates
df['quantity'] = listquantity


df.to_csv("resampled53665163_7397.csv")