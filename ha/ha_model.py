#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IEEE Transactions on Knowledge and Data Engineering, 2023.
#Predicting Irregular Mobility via Web Search Data-Driven Bipartite Graph Neural Networks
#WS-BiGNN
#Authors: Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma, and Satish V. Ukkusuri.


# In[2]:


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


# In[3]:


#model.run_model
#input1: x_u_v,            dim = 2
#input2: x_location        dim = (V, 200)
#input3: x_mobility_batch  dim = (batch, x_day, n_m, 2)
#input4: x_text_batch      dim = (batch, x_day, n_t, 2)
#input5: sorted_user       {'100018':0, '100451':1, ...}
#input6: sorted_location   {'10':0, '11':1, ...}
#input7: y_day             1
#output1: predict_score    dim = (batch, y_day, n_m, 2).

class HA(nn.Module):
    def __init__(self):
        super(HA, self).__init__()
    
    def run_model(self, x_u_v, x_location, x_mobility_batch, x_text_batch,                 sorted_user, sorted_location, y_day):
        batch, x_day = len(x_mobility_batch), len(x_mobility_batch[0])
        user_list, location_list = list(sorted_user), list(sorted_location)
        predict_score = [[0.0 for j in range(y_day)] for i in range(batch)]
            
        for i in range(batch):
            score_dict = {(user, location):0 for user in user_list                  for location in location_list}  
            
            for j in range(x_day):
                links = x_mobility_batch[i][j]   #(n_m, 2)
                for item in links:
                    user, location = item[0], item[1]
                    if user != "0" and location != "0":
                        score_dict[(user, location)] += 1         
            predict_score[i] = [score_dict for j in range(y_day)]
        return predict_score


# In[ ]:





# In[ ]:




