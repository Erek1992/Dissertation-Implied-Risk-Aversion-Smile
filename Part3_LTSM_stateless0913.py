# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 04:26:31 2021

@author: Erek
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 04:05:26 2021

@author: Erek
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:28:37 2021

@author: Erek
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:50:23 2021

@author: Erek
"""
###########################
#no stateful

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras import losses
tf.keras.backend.set_floatx('float64')



# alpha=tf.constant(2.)
# alpha=tf.reshape(alpha, [1, 1])

#first-step dataset
#Sim_Option = pd.read_csv("Sim_Option.csv")
Sim_Option=Sim_Option.sort_values(by=['C_P','strike','Date'],ascending=True,ignore_index=True)   
simdate=Sim_Option.copy()


simdate["St"]=np.log(simdate["St"])
simdate["strike"]=np.log(simdate["strike"])
simdate["Delta"]=0.
simdate=simdate.drop(columns=['Date', 'Rate','C_P',"Pt","P0",'exdate'])
simdate=np.array(simdate).reshape(26624,1,6)

P0=np.array(Sim_Option["P0"])
Pt=np.array(Sim_Option["Pt"])



class model:
    
    def __init__(self):
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.LSTM(64,kernel_initializer=xavier,activation=tf.nn.tanh,input_shape=[1,6], return_sequences=True)
        self.l2=tf.keras.layers.LSTM(32,kernel_initializer=xavier,activation=tf.nn.tanh,return_sequences = True)
        self.l3=tf.keras.layers.LSTM(16,kernel_initializer=xavier,activation=tf.nn.tanh)
        self.out=tf.keras.layers.Dense(1,kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adam( lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False  )
        
        

        
    # Running the model
    def run(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.l3(boom1)
        boom3=self.out(boom2)
        
        return boom3  
    
    
    def PNL(self,step0,step1,i):
        #print(simdate[i])
        s0=K.exp(step0[0][0][0])
        s0=tf.reshape(s0, [1, 1])
        d0=K.constant(step0[0][0][5])
        d0=tf.reshape(d0, [1, 1])
        d0=tf.cast( d0, dtype=tf.float64, name=None)
        s1=K.exp(step1[0][0])
        s1=tf.reshape(s1, [1, 1])
        d1=self.run(step0)
        Z=K.constant(Pt[index+1]-Pt[index])
        Z=tf.reshape(Z, [1, 1])
        Pzero=K.constant(P0[index])
        Pzero=tf.reshape(Pzero, [1, 1])
        H=d0*(s1-s0)
        C=K.abs(d1*s1-d0*s0)
        C=C*0.0005
        pnl1=tf.subtract(H, Z, name="pnl1")
        pnl2=tf.subtract(pnl1, C, name="pnl2")
        pnl3=tf.add(pnl2,Pzero,name="pnl3")
        #print("PNL",pnl3)
        return pnl3
        
    # def Utility(self,pnl):

    #     Ua= tf.exp(-0.2*pnl)
    #     return Ua

    def Utility(self,pnl):
        alpha=tf.constant(1.5)
        alpha=tf.reshape(alpha, [1, 1])
        alpha=tf.cast( alpha, dtype=tf.float64, name=None)
        # U=(1/alpha)*K.log(K.mean(K.exp(-alpha*pnl)))
        U=(1/alpha)*K.mean(K.exp(-alpha*pnl))
        
        #ut=(1-np.exp(-alpha[0][0].numpy()*pnl[0][0].numpy()))/alpha[0][0].numpy()
        Loss.append(U.numpy())
        return U
    
    def get_loss(self,step0,step1,i):
        pnl=self.PNL(step0,step1,i)
        loss=self.Utility(pnl)
        #print(loss)
        
        tloss = K.constant(loss)

        #print("Utility",loss)
        return K.mean(tloss)
    
    def get_grad(self,X,Y,i):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.l3.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(X,Y,i)
            g = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.l3.variables[0],self.l3.variables[1],self.out.variables[0],self.out.variables[1]])
        return g
      


        


    # perform gradient descent
    def network_learn(self,step0,step1,i):
        g = self.get_grad(step0,step1,i)
        self.train_op.apply_gradients(zip(g, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.l3.variables[0],self.l3.variables[1],self.out.variables[0],self.out.variables[1]]))


    


    
model=model()

# #
# model.network_learn(train,step1)
##################################################################################


# delta=[]
# step0=simdate[0]
# #for i in range(len(simdate)-1):
# for i in range(1000):
#     print(i," / ","max")
#     train=step0[:,np.newaxis,:]
    
#     step1=simdate[i+1]
#     #insert new delta
#     d1=model.run(train)
#     step1[0][5]=d1
#     #print(d1)
#     delta.append(d1.numpy()[0][0])
#     # pnl=model.PNL(train,step1)
#     # loss=model.Utility(pnl)
#     model.network_learn(train,step1)

#     step0=step1
        
     
        
#delta
        
        

index=0
Loss=[]

Sim_Option30=Sim_Option.copy()
Sim_Option30=Sim_Option30.drop(columns=['Rate'])
Sim_Option30['Delta']=0.   




for j in range(26):
    print(j+1,"/26")
    step0=simdate[j*1024]
#set the initial step
    Sim_Option30['Delta'][j*1024]=0.0
    for i in range(1023):
    #for i in range(1024,2047):
        #print(i)
        index=j*1024+i
        train=step0[:,np.newaxis,:]
        
        step1=simdate[index+1]
        #insert new delta
        d1=model.run(train)
        step1[0][5]=d1
        #print(d1)
        
        Sim_Option30['Delta'][index+1]=d1.numpy()[0][0]
    
        # loss=model.Utility(pnl)
        model.network_learn(train,step1,index)
    
        step0=step1
    print(sum(Loss))

Sim_Option30








# replication error
Err=[]

for j in range(26):
    step0=Sim_Option30.iloc[j*1024]
    p0=step0[7]
    pt=p0
    for i in range(1,1024):
        step1=Sim_Option30.iloc[j*1024+i]
        dt=step0[10]
        st=step0[1]
        st1=step1[1]
        pt1=step1[6]
        Err.append(np.abs(dt*(st1-st)+p0-(pt1-pt)))
        pt=pt1
        step0=step1
#print(Err)

plt.plot(Err[0:13299])
