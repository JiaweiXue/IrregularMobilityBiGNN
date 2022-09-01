#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IEEE Transactions on Knowledge and Data Engineering, 2023.
#Predicting Irregular Mobility via Web Search Data-Driven Bipartite Graph Neural Networks
#WS-BiGNN
#Authors: Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma, and Satish V. Ukkusuri.


# In[2]:


#Model_1: BiGNN(a,b) + Weight(d) + LongShortTerm(e) (WS_BiGNN)
#Model_2: BiGNN(a,b) + Weight(d) + LongShortTerm(e) + Hyperedges(c)  (WS_BiGNN_H)
#Model_3: BiGNN(a,b) + Weight(d) + LongShortTerm(e) + SuperNodes(Fig.3) (WS_BiGNN_S)
#Model_4: BiGNN(a,b) + Weight(d) + LongShortTerm(e) + Hyperedges(c) + SuperNodes(Fig.3) \
#(WS_BiGNN_H_S)

#WS_BiGNN. (1) User embedding; (2) GAT; (3) LongShortTerm; (4) DistMult
#WS_BiGNN_H. (1) User embedding; (2) GAT + Hyperedges; (3) LongShortTerm; (4) DistMult
#WS_BiGNN_S. (1) User embedding; (2) GAT; (3) LongShortTerm + SuperNodes; (4) DistMult


# In[3]:


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


# In[4]:


#input1: x_u_v,            dim = 2
#input2: x_location        dim = (V, 200)
#input3: x_mobility_batch  dim = (batch, x_day, n_m, 2)
#input4: x_text_batch      dim = (batch, x_day, n_t, 2)
#input5: sorted_user       {'100018':0, '100451':1, ...}
#input6: sorted_location   {'10':0, '11':1, ...}
#input7: y_day             1
#output1: predict_score    dim = (batch, y_day, n_m, 2)


# # 1: Define user embedding

# In[5]:


#functionality: define user embeddings from the location embeddings.
#input1: x_loation               dim = (V, 200)
#input2: x_mobility_batch        dim = (batch, x_day, n_m, 2)
#input3: x_text_batch            dim = (batch, x_day, n_t, 2)
#input4: sorted_user             {'100018':0, '100451':1, ...}
#input5: sorted_location         {'10':0, '11':1, ...}
#output: x_user                  dim = (batch, U, 200)
class User_embedding(nn.Module):
    def __init__(self):
        super(User_embedding, self).__init__()
        
    def forward(self, x_location, x_mobility_batch, x_text_batch, sorted_user, sorted_location):
        n_user, n_location = len(sorted_user), len(sorted_location)
        x_user = list()
        
        x_m_t_batch = torch.cat([x_mobility_batch, x_text_batch], dim=2)     ###!!!
        batch, x_day = x_m_t_batch.size()[0], x_m_t_batch.size()[1]
        #dim = (batch, x_day, n_m + n_t, 2)
        
        for i in range(batch):
            #step 1: initialize 
            user_average_embed = [[0.0 for loc in range(200)] for user in range(n_user)] ###!!!
            user_count_embed = [0 for user in range(n_user)] 
            user_loc_with_edge = list()
            
            #step 2: update the user embedding
            link_record = x_m_t_batch[i][0]
            for link in link_record:
                user_idx, loc_idx = sorted_user[link[0]], sorted_location[link[1]] 
                user_idx_with_edge.append(user_idx)              
                #update count 
                user_count_embed[user_idx] += 1       
                #update embedding
                n_count = user_count_embed[user_idx]
                user_average_embed[user_idx] = x_location[loc_idx]/(1.0*n_count) +                    user_average_embed[user_idx]*(n_count-1)/(1.0*n_count)
            set_user_idx_with_edge = set(user_idx_with_edge)
            n_user_with_edge = len(set_user_idx_with_edge)
            print ("# user with embedding: ", n_user_with_edge)
            print ("# n_user: ", n_user)
            
            #step 3: update the user embedding for other users
            #compute the average embedding     #200
            average_embedding =                torch.mean(torch.tensor(user_average_embed), dim=0)*n_user/(1.0*n_user_with_edge)
            
            #define the remaining embedding   
            for user_idx in range(n_user):
                if user_idx not in set_user_idx_with_edge:
                    user_average_embed[user_idx] = average_embedding
            x_user.append(user_average_embed)
        return x_user                                         ### !!!


# # 2. BiGraph Attention Network

# In[6]:


#2.1 
class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha, dropout=0.0, concat=True):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(input_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    #input1: feature.shape: (N, input_dim)
    #input2: adj.shape: (N, N)
    #output: h_prime.shape: (N, hidden_dim)
    def forward(self, feature, adjs):
        #feature.shape: (N, input_dim)
        #####print ("feature.shape", feature.shape)
        #####print ("W.shape", self.W.shape)
        #step 1: get Wh
        Wh = torch.mm(feature, self.W)    #Wh.shape: (N, hidden_dim)  
        
        #step 2: get e = LeakyReLU(a[Wh_{i};Wh_{j}])
        Wh1 = torch.matmul(Wh, self.a[:self.hidden_dim, :])  # Wh1&2.shape (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.hidden_dim:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)      # e.shape (N, N)
        
        #step 3: calculate the attention
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adjs > 0, e, zero_vec)  #attention.shape: (N, N)
        attention = F.softmax(attention, dim=1)         #attention.shape: (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        #step 4: get the h_prime
        h_prime = torch.matmul(attention, Wh) 
        if self.concat:
            return F.elu(h_prime)                      
        else:
            return h_prime
        
#2.2         
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha, dropout):
        super(GAT, self).__init__()
        self.attention = GATLayer(input_dim, hidden_dim, alpha, dropout, True)

    def forward(self, x, adj):
        x_out = F.relu(self.attention(x, adj))   
        return x_out


# In[7]:


#input1: x_mobility_batch   dim = (batch, x_day, n_m, 2)
#input2: x_text_batch       dim = (batch, x_day, n_t, 2)
#output1: x_adj             dim = (batch, x_day, U+2V, U+2V)
def convert_to_adj(x_mobility_batch, x_text_batch, sorted_user, sorted_location):
    batch, x_day = x_mobility_batch.size()[0], x_mobility_batch.size()[1]
    
    n_user, n_loc = len(sorted_user), len(sorted_location)
    adj_dim = n_user+2*n_loc
    adj = torch.zeros((batch, x_day, adj_dim, adj_dim))
    
    for i in range(batch):
        x_mobility_record, x_text_record = x_mobility_batch[i], x_text_batch[i]
        for j in range(x_day):
            x_mobility_one_day, x_text_one_day = x_mobility_record[j], x_text_record[j]
            #extract mobility edges
            for link in x_mobility_one_day:
                user_idx, loc_idx = sorted_user[link[0]], sorted_location[link[1]] 
                adj[i][j][user_idx][n_user + loc_idx] = 1
            #extract text edges
            for link in x_text_one_day:
                user_idx, loc_idx = sorted_user[link[0]], sorted_location[link[1]] 
                adj[i][j][user_idx][n_user + n_loc + loc_idx] = 1
    return adj, batch, x_day, n_user, n_loc


# In[8]:


#2.3 
#functionality: get user and location embeddings by propagating information along x days.
#input1: x_u_v                         dim = 2
#input2: x_user                        dim = (batch, U, 200)
#input3: x_location                    dim = (V, 200)
#input4: x_mobility_batch              dim = (batch, x_day, n_m, 2)
#input5: x_text_batch                  dim = (batch, x_day, n_t, 2)
#input5: sorted_user                    {'100018':0, '100451':1, ...}
#input6: sorted_location                {'10':0, '11':1, ...}
#output1: user_loc_embed_seq           dim = (batch, x_day, U+2V, 200)
class BiGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha, dropout):
        super(BiGNN, self).__init__()
        self.input_dim = input_dim
        self.user_embedding = User_embedding()
        self.gat = GAT(input_dim, hidden_dim, alpha, dropout)
        
    def forward(x_u_v, x_loc, x_mobility_batch, x_text_batch, sorted_user, sorted_loc):
        #step1：obtain user embedding (i.e., x_user)  
        x_user = self.user_embedding(x_loc, x_mobility_batch, x_text_batch,                                     sorted_user, sorted_loc)    #dim = (batch, U, 200).
        
        #step2: define adj          dim = (batch, x_day, U+2V, U+2V)
        #input1: x_mobility_batch   dim = (batch, x_day, n_m, 2)
        #input2: x_text_batch       dim = (batch, x_day, n_t, 2)
        #output1: x_adj             dim = (batch, x_day, U+2V, U+2V)
        #output2,3,4,5
        x_adj, batch, x_day, n_user, n_loc = convert_to_adj(x_mobility_batch,                                                      x_text_batch, sorted_user, sorted_loc)
        
        #step3: update the information  
        user_loc_embed_seq = torch.zeros((batch, x_day,                                                   n_user+2*n_loc, self.hidden_dim))
        #dim = (batch, x_day, U+2V, 200)
        
        #x_user dim = (batch, U, 200)   
        x_location = x_loc.repeat(batch, x_loc.size()[0], x_loc.size()[1])  #dim = (batch, V, 200)
        x_user_loc = torch.cat([x_user, x_location, x_location], dim=1)  #dim = (batch, U+2*V, 200)
        
        for i in range(batch):
            for j in range(x_day):
                if j == 0:
                    user_loc_embed_seq[i][j] = x_user_loc[i]
                else:
                    user_loc_embed_seq[i][j] =                        self.gat(user_loc_mob_text_embed_seq[i][j-1], x_adj[i][j-1])
        return user_loc_embed_seq


# # 3. LongShortTerm 

# In[9]:


#functionality: aggregate the embedding from the long and short distances.
#input1: user_loc_embed_seq             dim = (batch, x_day, U+2V, 200)
#output1: final_user_loc_embed_seq      dim = (batch, U+2V, 200)
class LST(nn.Module):
    def __init__(self, x_day):
        super(LST, self).__init__()
        self.a = nn.Parameter(torch.empty(size=x_day))
        nn.init.xavier_uniform_(self.a.data, gain=1.00)
        
    def forward(self, user_loc_embed_seq):
        input_size = user_loc_embed_seq.size()
        batch, x_day, u_2v, hidden_dim = input_size[0], input_size[1], input_size[2], input_size[3]
        final_user_loc_embed_seq = torch.zeros((batch, u_2v, hidden_dim))
        for i in batch:
            for j in x_day:
                final_user_loc_embed_seq[i] += self.a[j] * user_loc_embed_seq[i][j]
        return final_user_loc_embed_seq


# # 4. DistMult

# In[10]:


#functionality: predict the existence of links.
#input1: final_user_loc_embedding       dim = (batch, U+2V, 200)
#output: final link                     dim = (batch, U, V)
#predict_score ({('100018', '10'):0.2, ...})
class DistMult(nn.Module):
    def __init__(self, embed_dim, batch, n_user, n_loc):
        super(DistMult, self).__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        self.n_user = n_user
        self.n_loc = n_loc
    
    def forward(self, final_user_loc_embedding):
        #user_h (U, 200) * W (200, 200) * loc_mob_h (200, V)
        
        user_h = final_user_loc_embedding[:, :self.n_user, :]                  #(batch, U, 200)
        loc_mob_h = final_user_loc_embedding[:, self.n_user:self.n_user+self.n_loc, :]   #(batch, V, 200)
        user_h_W = torch.matmul(user_h, torch.diag(self.weight))          #(batch, U, 200)
        output_embed = torch.sigmoid(torch.matmul(user_h_W, loc_mob_embed.permute(0,2,1)))
        #dim = (batch, U, V)
        
        return output_embed


# # 5. WS_BiGNN

# In[11]:


#functionality: predict potential edges based on comment, mobility, and text data.
#input1: x_u_v             dim = 2
#input2: x_location        dim = (V, 200)
#input3: x_mobility_batch  dim = (batch, x_day, n_m, 2)
#input4: x_text_batch      dim = (batch, x_day, n_t, 2)
#input5: sorted_user       {'100018':0, '100451':1, ...}
#input6: sorted_location   {'10':0, '11':1, ...}
#input7: y_day             1
#output1: predict_score    dim = (batch, y_day, n_m, 2).
class WS_BiGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha, dropout, x_day, batch, n_user, n_loc):
        super(WS_BiGNN, self).__init__()
        self.bignn = BiGNN(input_dim, hidden_dim, alpha, dropout)
        self.ltm = LST(x_day)
        self.dist_mult = DistMult(hidden_dim, batch, n_user, n_loc)
        
    def run(self, x_u_v, x_loc, x_mobility_batch, x_text_batch,            sorted_user, sorted_loc, y_day):
        
        user_loc_embed_seq =            self.bignn(x_u_v, x_loc, x_mobility_batch, x_text_batch, sorted_user, sorted_loc)
        #dim = (batch, x_day, U+2V, 200)
        
        final_user_loc_embed_seq = self.LST(user_loc_embed_seq) #dim = (batch, U+2V, 200)

        output_embed = self.dist_mult(final_user_loc_embed_seq)
        output_embed = output_embed.unsqueeze(dim=1).repeat((1,y_day,1,1))
        return output_embed


# In[ ]:





# In[ ]:




