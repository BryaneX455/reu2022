# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:15:30 2022

@author: Bryan
"""
import numpy as np
from GHM_Classes.GHM import GHM
from GHM_Classes.GHM_Datagen import Data_Gen






Num_Sample = 10000
Num_Node = 25
Kap = 5
ItNum = 30
s = np.random.randint(5, size=1*Num_Node)
"""
Data Generation All State NWS
"""
knn = 4
prob = 0.65
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=prob, knn=knn, kap=Kap, GHMItNum = ItNum, s = s)
df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam =  Data_Gen_Class.Sample_NWSGHM()
df.to_csv('GHM_Dict_Data_NWS_AllStates.csv')  
print('NWS:', 'SyncNum:', SyncNum, 'NonSync:', NonSync, 'Ave_Edge:', Ave_Edge, 'Std_Edge:', Std_Edge, 'Ave_Diam:', Ave_Diam, 'Std_Diam:', Std_Diam)



"""
Data Generation All State UCLA26
"""

Num_Sample = 10000
Num_Node  = 25
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum, s = s)
df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam =  Data_Gen_Class.Sample_UCLAGHM()
df.to_csv('GHM_Dict_Data_UCLA26_AllStates.csv')  
print('UCLA:', 'SyncNum:', SyncNum, 'NonSync:', NonSync, 'Ave_Edge:', Ave_Edge, 'Std_Edge:', Std_Edge, 'Ave_Diam:', Ave_Diam, 'Std_Diam:', Std_Diam)


"""
Data Generation All State Caltech36
"""

Num_Sample = 10000
Num_Node  = 25
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum, s = s)
df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam =  Data_Gen_Class.Sample_CaltechGHM()
df.to_csv('GHM_Dict_Data_Caltech36_AllStates.csv')  
print('Caltech:', 'SyncNum:', SyncNum, 'NonSync:', NonSync, 'Ave_Edge:', Ave_Edge, 'Std_Edge:', Std_Edge, 'Ave_Diam:', Ave_Diam, 'Std_Diam:', Std_Diam)



"""
Data Generation All State Harvard1
"""
"""
Num_Sample = 10000
Num_Node  = 30
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum, s = s)
df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam =  Data_Gen_Class.Sample_HarvardGHM()
df.to_csv('GHM_Dict_Data_Harvard1_AllStates.csv')  
print('Harvard:', 'SyncNum:', SyncNum, 'NonSync:', NonSync, 'Ave_Edge:', Ave_Edge, 'Std_Edge:', Std_Edge, 'Ave_Diam:', Ave_Diam, 'Std_Diam:', Std_Diam)
"""


"""
Data Generation All State Wisconsin87
"""
"""
Num_Sample = 10000
Num_Node  = 30
Kap = 5
ItNum = 30
Data_Gen_Class = Data_Gen(num_samples = Num_Sample, NodeNum = Num_Node, prob=None, knn=None, kap=Kap, GHMItNum = ItNum, s = s)
df, SyncNum, NonSync, Ave_Edge, Std_Edge, Ave_Diam, Std_Diam =  Data_Gen_Class.Sample_WisconsinGHM()
df.to_csv('GHM_Dict_Data_Wisconsin87_AllStates.csv')  
print('Wisconsin', 'SyncNum:', SyncNum, 'NonSync:', NonSync, 'Ave_Edge:', Ave_Edge, 'Std_Edge:', Std_Edge, 'Ave_Diam:', Ave_Diam, 'Std_Diam:', Std_Diam)
"""