# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:11:07 2021

@author: Erek
"""
import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scipy
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import tensorflow.keras.backend as K




def BS(S0, strike, T, sigma,r,n):
    d1 =  n*(np.log(S0/strike)+T*(r+0.5*sigma**2))/(np.sqrt(T)*sigma)
    d2 =  n*((np.log(S0/strike)+T*(r+0.5*sigma**2))/(np.sqrt(T)*sigma) - sigma * np.sqrt(T))
    Nd1 = scipy.norm.cdf(d1)
    Nd2 = scipy.norm.cdf(d2)
    # call option
    if n == 1:
        p = S0 * Nd1 - strike * Nd2
    #put option
    else:
        p = strike * Nd2 - S0 * Nd1  
    return p

def Delta(S0, strike, T, sigma,r,n):
    Nd1 =  scipy.norm.cdf((np.log(S0/strike)+T*(r+0.5*sigma**2))/(np.sqrt(T)*sigma))
    if n == 1:
        delta = Nd1
    else:
        delta = Nd1 - 1
    return delta


# set real calendat
calendar=pd.bdate_range(start="20100104",end="20201231")

# Import historical asset price
Spot = pd.read_csv("Spot price with sigma.csv")

#Compute daily return with shuffled 30-day chucks 
diff=[]
for i in range(len(Spot)-1):
    diff.append(Spot["close"][i+1]/Spot["close"][i])


chucks= []
for j in range(92):
    chucks+= [diff[j*30:j*30+30]]

np.random.shuffle(chucks)

diff_histroy_shuffle_chuck=[100]
for I in range(92):
    for J in range(30):
        a = chucks[I][J]*diff_histroy_shuffle_chuck[-1]
        diff_histroy_shuffle_chuck.append(a)
        
# Set S0=100
diff_histroy_shuffle_chuck_add29=[]

for i in range(30):
    diff_histroy_shuffle_chuck_add29.append(chucks[20][i])

chucks_new=[100.0]
for i in range(29):
    chucks_new.append(diff_histroy_shuffle_chuck_add29[i]*chucks_new[-1])
    
chucks_new+=diff_histroy_shuffle_chuck    

plt.plot(chucks_new[0:1024])
plt.ylabel('Asset Price')
plt.xlabel('Time step')
plt.title('Simulated Price Path')
plt.show()

logreturns=np.diff(np.log(chucks_new))

#compute simulated volatility
volatility = pd.Series(logreturns).rolling(window=30).std()*np.sqrt(252)

SimSt = chucks_new[1:len(chucks_new)]
SimSt = np.array(SimSt[29:len(SimSt)])


LRate = pd.read_csv("Libor rate.csv")
LRate = LRate[~LRate["USDONTD156N"].isin(["."])]
LRate = np.array(LRate["USDONTD156N"],dtype=np.float)



#Calculate P0 with BS
SimVt = volatility[29:len(volatility)]
SimVt[29]=0.2
SimVt=np.array(SimVt)
K0 = [70.0+i*5 for i in range(13)]
CP0=[]
PP0=[]
for i in K0:
    CP0.append(BS(S0=100, strike=i, T=1, sigma=0.2,r=0,n=1))
    PP0.append(BS(S0=100, strike=i, T=1, sigma=0.2,r=0,n=-1))

	

#Combine information together 
Sim_Option=pd.DataFrame()
temp1=pd.DataFrame(columns={"Date":"","St":"","Rate":"","strike":"","Vt":"","C_P":"","Pt":"","P0":"","Pt":""},index=[0],dtype=(np.float))
temp2=pd.DataFrame(columns={"Date":"","St":"","Rate":"","strike":"","Vt":"","C_P":"","Pt":"","P0":"","Pt":""},index=[0],dtype=(np.float))



for i in range(1024):
    for j in range(len(K0)):
        temp1["Date"][0] = calendar[i]
        temp1["strike"][0]=K0[j]
        temp1["St"][0]=SimSt[i]
        temp1["Rate"][0]=LRate[i]
        temp1["Vt"][0]=SimVt[i]
        temp1["C_P"][0]="C"
        temp1["P0"][0]=CP0[j]
        temp1["PK"]=temp1["St"]-temp1["strike"]
        Sim_Option =Sim_Option.append(temp1,ignore_index=True)
        
        temp2["Date"][0] = calendar[i]
        temp2["strike"][0]=K0[j]
        temp2["St"][0]=SimSt[i] 
        temp2["Rate"][0]=LRate[i]
        temp2["Vt"][0]=SimVt[i]
        temp2["C_P"][0]="P"
        temp2["P0"][0]=PP0[j]
        temp2["PK"]=temp2["strike"]-temp2["St"]
        Sim_Option =Sim_Option.append(temp2,ignore_index=True)
        
Sim_Option["exdate"]=pd.to_datetime(20131205,format="%Y%m%d")
Sim_Option["TTM"]=Sim_Option['exdate']-Sim_Option["Date"]
Sim_Option['TTM']= pd.to_numeric(Sim_Option['TTM'])/86400000000000/365       
#Sim_Option['PK'] = Sim_Option['PK']*np.exp(-Sim_Option['TTM']*Sim_Option['Rate'])
Sim_Option = Sim_Option.sort_values(by=['C_P','strike','Date',],ascending=True)



#compute Pt with Gaussian regressor 
Sim_c = Sim_Option.copy()
Sim_c = Sim_c[Sim_c['C_P']=='C']


Sim_p = Sim_Option.copy()
Sim_p = Sim_p[Sim_p['C_P']=='P']

vol = np.array(volatility[29:2791])
spot = np.array(SimSt[0:2761])
cP0 = pd.unique(Sim_c['P0'])
pP0 = pd.unique(Sim_p['P0'])


#Call Options
kernel = C(0.1, (1,20))*RBF(35,(1e-5,35))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.1)
obs = np.array([[70,0.2,100,cP0[0]],[75,0.2,100,cP0[1]],[80,0.2,100,cP0[2]],[85,0.2,100,cP0[3]], \
                [90,0.2,100,cP0[4]],[95,0.2,100,cP0[5]],[100,0.2,100,cP0[6]],[105,0.2,100,cP0[7]], \
                [110,0.2,100,cP0[8]],[115,0.2,100,cP0[9]],[120,0.2,100,cP0[10]],[125,0.2,100,cP0[11]],[130,0.2,100,cP0[12]]])
        #fit gp
gp.fit(obs[:,:-1],obs[:,-1])


call_price = []   
for i in range(0,len(vol)):
#    sigma = vol[i]
        
            
        #set obs_set
    xset, yset, zset = np.arange(70, 135, 5), vol[i] * np.ones(13), spot[i]*np.ones(13)
#        xset, yset = np.array[70], np.array[0.2]
        
        #predict
    obs_test = np.c_[xset.ravel(), yset.ravel(),zset.ravel()]
    outputs= gp.predict(obs_test)
    call_price.append(outputs)
        
option_price_c = np.array(call_price)

#############
#Put Options
kernel1 = C(0.1, (1,20))*RBF(35,(1e-5,35))
gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=0.1)
obs1 = np.array([[70,0.2,100,pP0[0]],[75,0.2,100,pP0[1]],[80,0.2,100,pP0[2]],[85,0.2,100,pP0[3]], \
                [90,0.2,100,pP0[4]],[95,0.2,100,pP0[5]],[100,0.2,100,pP0[6]],[105,0.2,100,pP0[7]], \
                [110,0.2,100,pP0[8]],[115,0.2,100,pP0[9]],[120,0.2,100,pP0[10]],[125,0.2,100,pP0[11]],[130,0.2,100,pP0[12]]])
        #fit gp
gp1.fit(obs1[:,:-1],obs1[:,-1])


put_price = []   
for i in range(0,len(vol)):

                  
        #set obs_set
    xset1, yset1,zset1 = np.arange(70, 135, 5), vol[i] * np.ones(13), spot[i]*np.ones(13)
#        xset, yset = np.array[70], np.array[0.2]
        
        #predict
    obs1_test = np.c_[xset1.ravel(), yset1.ravel(),zset1.ravel()]
    outputs, err = gp1.predict(obs1_test,return_std=True)
    put_price.append(outputs)
        
option_price_p = np.array(put_price)
#Change to 1D np.array
option_price_c = option_price_c.ravel()
option_price_p = option_price_p.ravel()


#Merge the option price with Sim_Option_c
P_call = pd.DataFrame(option_price_c)
Sim_c = Sim_c.sort_values(by=['Date','strike','C_P',],ascending=True,ignore_index=True)   
Sim_c['Pt']=P_call      

P_put = pd.DataFrame(option_price_p)
Sim_p = Sim_p.sort_values(by=['Date','strike','C_P',],ascending=True,ignore_index=True)   
Sim_p['Pt']=P_put   


Sim_Option=Sim_c.append(Sim_p)
Sim_Option = Sim_Option.sort_values(by=['Date','strike','C_P'],ascending=True,ignore_index=True)   


Sim_Option["Delta"]=0.

#compute Delta
# for i in range(26,len(Sim_Option)):
#     S = Sim_Option['St'][i]
#     K = Sim_Option['strike'][i]
#     T = Sim_Option['TTM'][i]
#     sigma = Sim_Option['Vt'][i]
#     r = Sim_Option['Rate'][i]
#     if Sim_Option['C_P'][i] == 'C':
#         Sim_Option['Delta'][i] = Delta(S, K, T, sigma, r, 1)
#     elif Sim_Option['C_P'][i] == 'P':
#         Sim_Option['Delta'][i] = Delta(S, K, T, sigma, r, -1)








Sim_Option

Sim_Option=Sim_Option.sort_values(by=['C_P','strike','Date'],ascending=True,ignore_index=True)   

#Sim_Option.to_csv('C:/Users/Erek/Disssertation/Sim_Option.csv')

