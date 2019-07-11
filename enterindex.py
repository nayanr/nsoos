# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:11:41 2019

@author: n0r00te
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:06:17 2019

@author: n0r00te
"""

import requests
import csv


types = [ "inventory_quantity","inventory_distributorId" ,"orderNo","omsCallDate", "inventory_offerId","orderDate","inventory_distributorType","inventory_groupTransactionId",
                "inventory_environmentId",
                "inventory_createdBy",
                "inventory_createdDate",
                "type"]


    
index = "v2-trace-2019-06-28"
      
with open(index +'.csv', 'w') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(types)
    url1 = "https://elastic-search-prod-sse-logging-db-samsclubdotcom.southcentralus.cloudapp.azure.com:1433/" + index + "/_search"
    payload = { "from" : 0, "size" : 0, "query" : { "bool" : { "filter" : [ { "term" : { "type.keyword" : "inventory" } }] } } }
    req = requests.get(url1,json = payload,verify = False)
    length =  req.json()['hits']['total']
    print(length)
    print(int(length/10000)+1)
    print(length%10000)
    j = 0
    for i in range(0,int(length/10000)+1):
        
        payload = { "from" : i*10000, "size" : 10000, "query" : { "bool" : { "filter" : [ { "term" : { "type.keyword" : "inventory" } }] } } }
        r = requests.get(url1, json= payload, verify=False)
        for document in r.json()['hits']['hits']:
            toadd = []
            for type in types:
                if type in document['_source']:
                    toadd.append(document['_source'][type])
                else:
                    toadd.append('')
            writer.writerow(toadd)
        j = i 
        
    print(j)
        
    
    payload = { "from" : j*10000, "size" : length%10000, "query" : { "bool" : { "filter" : [ { "term" : { "type.keyword" : "inventory" } }] } } }
    r = requests.get(url1, json= payload, verify=False)
    for document in r.json()['hits']['hits']:
        toadd = []
        for type in types:
            if type in document['_source']:
                toadd.append(document['_source'][type])
            else:
                toadd.append('')
        writer.writerow(toadd)    

        
        
    

