# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:15:30 2022

@author: Bryan
"""

from GHM_Classes.GHM import GHM
from GHM_Classes.GHM_WS_Datagen_AllStates import Data_Gen

"""
Data Generation All State WS
"""
"""
Num_Sample = 3000
Num_Node = 16
Kap = 5
ItNum = 30
knn = 4
prob = 0.65
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=prob, knn=knn, kap=Kap, GHMItNum = ItNum)
df,SyncNum =  Data_Gen_Class.Equal_Sample_WSGHM()
df.to_csv('GHM_Dict_Data_AllStates.csv')  
print(SyncNum)
"""

"""
Data Generation All State UCLA26
"""
"""
Num_Sample = 1000
Num_Node  = 20
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum)
df,SyncNum =  Data_Gen_Class.Equal_Sample_UCLAGHM()
df.to_csv('GHM_Dict_Data_UCLA26_AllStates.csv')  
print(SyncNum)
"""

"""
Data Generation All State Caltech26
"""
"""
Num_Sample = 1000
Num_Node  = 20
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum)
df,SyncNum =  Data_Gen_Class.Equal_Sample_CaltechGHM()
df.to_csv('GHM_Dict_Data_Caltech36_AllStates.csv')  
print(SyncNum)"""


"""
Data Generation All State Harvard1
"""
"""
Num_Sample = 1000
Num_Node  = 20
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum)
df,SyncNum =  Data_Gen_Class.Equal_Sample_HarvardGHM()
df.to_csv('GHM_Dict_Data_Harvard1_AllStates.csv')  
print(SyncNum)
"""

"""
Data Generation All State Wisconsin87
"""

Num_Sample = 1000
Num_Node  = 20
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum)
df,SyncNum =  Data_Gen_Class.Equal_Sample_WisconsinGHM()
df.to_csv('GHM_Dict_Data_Wisconsin87_AllStates.csv')  
print(SyncNum)