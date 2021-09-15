# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 00:43:18 2021

@author: Erek
"""



HistoricalData = SP4.sort_values(by=["strike_price",'optionid','exdate','date'],ascending=True,   ignore_index=True)
number = pd.unique(HistoricalData['optionid'])


#HistoricalData = SP4.sort_values(by=["strike_price",'optionid','exdate','date'],ascending=True,   ignore_index=True)


#length=100000
length=len(HistoricalData)


temp=HistoricalData.iloc[0:length]
hisdata1=pd.DataFrame()
hisdata1["St"]=np.log(temp["close"])
hisdata1["strike"]=np.log(temp["strike_price"])
hisdata1["Vt"]=temp["sigma"]
hisdata1["SK"]=temp["SK"]
hisdata1["TTM"]=temp["TTM"]
hisdata1["Delta"]=0.

temp["best_offer"]=temp["best_offer"][0]
P0=np.array(temp["best_offer"])
Pt=np.array(temp["best_bid"])
#Option ID + exdate + date

hisdata1=np.array(hisdata1).reshape(length,1,6)

temp["Delta"]=0.0


# train NN with historical data

#set the initial step
step0=hisdata1[0]
for i in range(len(hisdata1)-1):

    print(i)
    index=i
    train=step0[:,np.newaxis,:]
    
    step1=hisdata1[i+1]

    d1=model.run(train)
    step1[0][5]=d1

    temp["Delta"][i+1]=d1.numpy()[0][0]

    model.network_learn(train,step1,index)
    step0=step1
    

# Fit NN with historical data
step0=hisdata1[0]
for i in range(len(hisdata1)-1):
#for i in range(1024,2047):
    #print(i)
    train=step0[:,np.newaxis,:]
    step1=hisdata1[i+1]
    #insert new delta
    d1=model.run(train)
    step1[0][5]=d1
    temp["Delta"][i+1]=d1.numpy()[0][0]
    step0=step1










# replication error
temp['date']=pd.to_numeric(temp['date'])/86400000000000
Err=[]
step0=temp.iloc[0]
p0=step0[5]
pt=p0
for i in range(len(temp)-1):

    step1=temp.iloc[i+1]
    dt=step0[15]
    st=step0[8]
    st1=step1[8]
    pt1=step1[4]
    rf=step0[9]
    time= step1[0]-step0[0]
    time=time/365
    
    Err.append(np.abs(dt*(st1-st)+p0*np.exp(rf*time)-(pt1-pt)))
    pt=pt1
    step0=step1
#print(Err)

plt.plot(Err)








