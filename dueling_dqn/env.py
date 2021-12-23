import gym
from Experience_pool import Experience_pool
#from q_network import q_network
import numpy as np
import torch
class environment:
    def __init__(self,env_name='CartPole-v0'):
        self.env = gym.make(env_name)
        self.reset()
    
    #restart the environment
    def reset(self):
        #reset the env, and return the initial state
        self.done = False
        self.current_state = torch.tensor(self.env.reset(),dtype=torch.float32)
        #self.env.render()
    
    #the success function for modeling the interaction between agent and evn, return a experience of the state transition
    #a episode contain (state, reward, action, next state, done, info)
    def react(self,action):
        #execute action get respond from the env
        observation, reward, done, info = self.env.step(action)
        observation = torch.tensor(observation,dtype=torch.float32)
        reward = reward if not done else -20
        self.current_state = observation
        self.done = done
        done = 1 if done else 0
        return  reward, observation, done
    
    #get current state info
    def get_current_state(self):
        return self.current_state

    

    #shape of action shape
    def get_action_space(self):
        return self.env.action_space.n
    #shape of observation shape
    def get_observation_space(self):
        return self.current_state.shape
    
    #is game end
    def isepisode_done(self):
        return self.done
"""   
if __name__ == "__main__":
    s = environment()
    print(s.react(1))
    print(s.current_state)
    #testing experience pool and env
    #experience = Experience_pool(32,100)
    #for i in range(32):
        #transition = s.react_to_action(1)
        #s.reset()
        #experience.store_experience(transition)
    #testing q_network
    #Q1 = q_network(s.get_action_space(),s.get_observation_space(),32)
    #Q2 = q_network(s.get_action_space(),s.get_observation_space(),32)
    #Q1.train(experience.sample_experience(),0.9,Q2)
"""
