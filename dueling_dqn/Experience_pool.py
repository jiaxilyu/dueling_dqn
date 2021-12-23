
from collections import namedtuple,deque
import numpy as np
import random


class Experience_pool:
    def __init__(self,batch_size,buffer):
        self.buffer = buffer
        self.batch_size = batch_size
        #need to set up experience pool
        self.experiences = deque(maxlen=buffer)
        #s: current state, target: target value, r : reward, done : finish or not, action: choosing action
        #target is the model prediction for the current state, s_ is max q value of next state
        self.Episode = namedtuple('transition',['s','r','a','s_','done'])
    
    #check whether the pool is full or not
    def check_pool_full(self):
        return len(self.experiences) >= self.buffer

    #a episode contain (state, reward, action, next_state_q_a, done, info)
    #episode is tuple, need to change it to named tuple
    def store_experience(self,transition):
        #convert tuple as named tuple
        transition = self.Episode(transition[0],transition[1],transition[2],transition[3], transition[4])
        self.experiences.append(transition)
    
    #return mini-batch
    def sample_experience(self):
        return random.sample(self.experiences,self.batch_size)
