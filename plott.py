# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:57:17 2019

@author: n0r00te
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt
data = pd.read_csv('all.csv')


print(data.inventory_offerId.unique().size)




print(data.shape)


offerid = 51717556


df = data.loc[data['inventory_offerId'] == offerid ]


print(df['inventory_distributorId'].value_counts())
list = df.inventory_distributorId.unique()
print(df.inventory_distributorId.unique())


fig = plt.figure()



for item in list:
    inventoryid = item
    dfi = df.loc[df['inventory_distributorId'] == inventoryid]
    dfi['orderDate'] = pd.to_datetime(dfi['orderDate'])
    dfi = dfi.astype({"inventory_quantity": int})
    dfi = dfi.sort_values(by=['orderDate'],ascending=True)
    
    
    ax = plt.gca()
    #ax = ax.xaxis.set_tick_params(rotation=30)
    string = "inventory_id " + inventoryid + " for " + "offerid " + str(offerid)  
    plt.title("")
    dfi.plot(kind='line',x='orderDate',y='inventory_quantity',label = string,ax=ax)
    
    plt.show()
    
    
    
    
    
    
    
    
    
    






