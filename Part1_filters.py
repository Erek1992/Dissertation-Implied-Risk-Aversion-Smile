# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd

# import dataset
#SP = pd.read_csv("Data.csv")
SP = pd.read_csv("Data1920.csv")

SP= SP.drop(columns=['forward_price'])
SP= SP.drop(columns=['issuer','exercise_style','index_flag'])

S = pd.read_csv("Spot price with sigma.csv")

## compute close diff
# contractData=np.array(S["close"])
# contractData
# nDays=len(contractData)
# nDays
# priceArray=contractData#.tolist()
# lnPriceArray= np.log(priceArray)
# diffPriceArray=[0]
# #for i in range(1,len(nDays)):
# for i in range(1,nDays):
#     diffPriceArray.append(lnPriceArray[i]-lnPriceArray[i-1]);
# diffPriceArray
# diff=np.array(diffPriceArray)
# S["diff"]=diff


SP= pd.merge(SP,S[['date','close','sigma']],how = 'inner',on = ['date'])


SP['date']= pd.to_datetime(SP['date'],format="%Y%m%d")
SP['exdate']= pd.to_datetime(SP['exdate'],format="%Y%m%d")

SP['TTM']=SP['exdate']-SP['date']
SP['TTM']= pd.to_numeric(SP['TTM'])/86400000000000/365


Rf = pd.read_csv("Libor rate.csv")
Rf["date"]=pd.to_datetime(Rf["date"],format="%d/%m/%Y")
Rf=Rf.rename(columns={'USDONTD156N':'RF'}) 
Rf['RF']= pd.to_numeric(Rf['RF'], errors='coerce').fillna(method="ffill")
#Rf

SP= pd.merge(SP,Rf,how = 'left',on = ['date'])
F = pd.read_csv("Forward Price1.csv")
F['exdate']=F['expiration']
SP

#We filtered out options with a mid-price (0:5 * (bid price + ask price)) below 3/8
SP['mid_price'] = 0.5*(SP['best_bid']+SP["best_offer"])

#filter1: remove options where its volatility more than 1.
SP1 = SP[SP['impl_volatility'] <= 1]
#SP1_uniid = np.unique(fliter1['optionid'])
#SP1 = SP[~SP['optionid'].isin(list(SP1_uniid))]

SP2 = SP1[SP1['mid_price'] >= 3/8]
# SP2_uniid = np.unique(fliter2['optionid'])
# SP2 = SP1[~SP1['optionid'].isin(list(SP2_uniid))]

#removed negative and zero price
SP3 = SP2[SP2['best_bid'] > 0]
#SP3_uniid = np.unique(filter3['optionid'])
#SP3 = SP2[~SP2['optionid'].isin(list(SP3_uniid))]
SP3['strike_price'] = SP3['strike_price']/1000

SP3['PVK'] = SP3['strike_price']*np.exp(-SP3['TTM']*SP3['RF'])


SP3["m"] = SP3["close"]-SP3["PVK"]
SP3["m"] = np.where(SP3["m"] <0,0,SP3["m"])

SP3["n"] = -(SP3['PVK']-SP3['close'])
SP3["n"] = np.where(SP3["n"] <0,0,SP3["n"])

c = SP3[SP3['cp_flag'] == 'C']
#c = c[(c['mid_price']>=c['close']) | (c["m"]>=c['mid_price'])]
c1 = c[(c['mid_price']<=c['m']) | (c["mid_price"]>=c['close'])]
p = SP3[SP3['cp_flag'] == 'P']
p1 = p[(p['mid_price']<=p['n']) | (p["mid_price"]>=p['PVK'])]
SP4 = c1.append(p1)
SP4 = SP4.sort_values(by=['optionid','cp_flag','exdate','date'],ascending=True)


SP4= SP4.drop(columns=['m','n'])


# SP4["opdiff"]=1

# Id = np.unique(SP4["optionid"])

# SP5=pd.DataFrame()


# for i in Id:
#     SP4.loc[SP4.optionid == i] 
#     for j in range(1,len(n)):
#         n.iloc[j,19]=n.iloc[j,4]-n.iloc[j-1,4]        
#     SP5= SP5.append(n)
    
#SP4.to_csv('C:/Users/Erek/Disssertation/SPX0819.csv')

########################################################################



#V_t


# def ComputeVolatility (contractData):    
# #    //包含多少天的标的合约价格    
#     nDays=contractData.length 

#  #   //获取每日收盘价（或者结算价）并存入数组 
#     priceArray=contractData.close 
    
# #   //对价格取自然对数    
#     lnPriceArray=[ln(x) for x in priceArray] 

# #   //以下表示取对数价格的差，并存在diffPriceArray数组中，   //我们忽略了边界条件，实际  得到数组长度为nDays-1
#     for i in range(nDays)：
#         diffPriceArray[i]=lnPriceArray[i]-lnPriceArray[i-1]
   
# #   //计算波动率   
#    sigma=standard_deviation(diffPriceArray) * sqrt(250/nDays)
#    return sigma





# S2 = pd.read_csv("spotprice.csv")

# # compute close diff
# contractData=np.array(S2["close"])
# contractData

# nDays=len(contractData)
# nDays

# priceArray=contractData#.tolist()

# lnPriceArray= np.log(priceArray)

# diffPriceArray=[0]

# #for i in range(1,len(nDays)):
# for i in range(1,nDays):
#     diffPriceArray.append(lnPriceArray[i]-lnPriceArray[i-1]);

# diffPriceArray

# diff=np.array(diffPriceArray)


# S2["diff"]=diff

# S2["sd"]= np.std(S2["diff"]) * np.sqrt(250/nDays)
###
# SP4 = SP3 
# SP4['test'] = SP4[(SP4['mid_price']>=SP4['close']) | (SP4["m"]>=SP4['mid_price'])]

# SP4
# SP4['test']


import tensorflow.keras.backend as K


def Entropy(wealth=None, w=None, loss_param=None):
    _lambda = loss_param

#Entropy (exponential) risk measure
    return (1/_lambda)*K.exp(-_lambda*wealth)