from model import dueling_q_network
from collections import OrderedDict 
import numpy as np
import torch


#only for choosing action
class agent:
    #gamma is the insight of the agent, how it weight the future reward
    def __init__(self,action_space,epsilon,network):
        self.action_space = action_space
        #exploration rate
        self.epsilon = epsilon
        #subscribe the action brain
        self.brain = network
    
    #require action
    def take_action(self,observation):
        #get observation from q approximation
        prediction = self.brain(observation)
        #------------choosing action-----------
        #take greedy action
        if np.random.random_sample(1)[0] > self.epsilon:
            #print(prediction)
            action = torch.argmax(prediction).item()
            #action_value = prediction[action]
        #random choosing action
        else:
            action = np.random.randint(0,self.action_space)
            #action_value = prediction[action]
        return action
    
        
        
