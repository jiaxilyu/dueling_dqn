from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class q_network(nn.Module):
    def __init__(self,action_space,state_space):
        super(q_network, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.weight_init = nn.init.xavier_uniform
        #common network(cnn)
        
        self.l1 = nn.Linear(state_space,24)
        self.weight_init(self.l1.weight)
        self.af1 = nn.ReLU()
        self.l2 = nn.Linear(24,24)
        self.weight_init(self.l2.weight)
        self.af2 = nn.ReLU()
        self.l3 = nn.Linear(24,24)
        self.weight_init(self.l3.weight)
        self.af3 = nn.ReLU()
        self.output = nn.Linear(24,self.action_space)
    #build up the initial model and initialize it
    def forward(self,x):
        x = self.l1(x)
        x = self.af1(x)
        x = self.l2(x)
        x = self.af2(x)
        r = self.output(x)
        #adv_substract = adv - adv.mean(dim=0, keepdim=True)
        return r
        #return v + adv_substract
    

class dueling_q_network(nn.Module):
    def __init__(self,action_space,state_space):
        super(dueling_q_network, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.weight_init = nn.init.xavier_uniform
        
        #common network
        self.l1 = nn.Linear(state_space,64)
        self.weight_init(self.l1.weight)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64,64)
        self.weight_init(self.l2.weight)
        
        #value network
        self.fc_v = nn.Linear(64,256)
        self.weight_init(self.fc_v.weight)
        self.v = nn.Linear(256,1)
        
    
        #adv
        self.fc_adv = nn.Linear(64,256)
        self.weight_init(self.fc_adv.weight)
        self.adv = nn.Linear(256,self.action_space)
    #build up the initial model and initialize it
    def forward(self,x):
         
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        
        v = self.fc_v(x)
        v = self.relu(v)
        v = self.v(v)

        adv = self.fc_adv(x)
        adv = self.relu(adv)
        adv = self.adv(adv)
        adv_substract = adv - adv.mean(dim=0, keepdim=True)
        
        return v + adv_substract


