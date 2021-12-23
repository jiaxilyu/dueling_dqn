from os import stat
import gym
from Experience_pool import Experience_pool
from model import dueling_q_network, q_network
import numpy as np
from env import environment
from agent import agent
import matplotlib.pyplot as plt
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing

#epsilon
epsilon = 1
#mini batch size
batch_size = 128
#future reward weight
gamma = 0.95
#control the ratio of action brain and target brain in the soft update
tau = 0.95
#maximum trainnig steps
max_episodes = 800
buffer = 3000
#how many steps per update
start_training_steps = 300
#how many steps per update
update_steps = 4
#game env
env_name = 'CartPole-v0'
LOSS_FN =  nn.MSELoss()
OPTIMIZER = torch.optim.Adam

#fit the experience
def computed_td_loss(b_model, t_model,experiences_sample):
        s_stack = torch.stack([experience.s for experience in experiences_sample],dim=0)
        a_stack = [experience.a for experience in experiences_sample]
        #a_stack = torch.reshape(a_stack,(len(a_stack), 1))
        s_next_stack = torch.stack([experience.s_ for experience in experiences_sample])
        r_stack = torch.tensor([experience.r for experience in experiences_sample])
        #r_stack = torch.reshape(r_stack,(len(r_stack), 1))
        done_stack = torch.tensor([experience.done for experience in experiences_sample])
        #compute the next state max Q(s_,a)
        s_next_pred = t_model(s_next_stack).max(-1)[0].detach()
        target_v = r_stack + (1-done_stack)*s_next_pred*gamma
        #print((1-done_stack)*s_next_pred*gamma)
        #compute the target vector
        s_pred = t_model(s_stack)[range(len(s_next_pred)), a_stack]
        return LOSS_FN(s_pred,target_v.detach())
    
class dqn_agent:
    def __init__(self,gamma = gamma):
        self.agent_name = "dqn"
        #training config setting
        #mini batch size
        self.batch_size = batch_size
        #future reward weight
        self.gamma = gamma
        #control the ratio of action brain and target brain in the soft update
        self.tau = np.float32(tau)
        #maximum trainnig steps
        self.max_episodes = max_episodes
        self.total_steps = 0
        self.episodes = 0
        self.buffer = buffer
        
        #how many steps per update
        self.update_steps = update_steps
        #the steps start training
        self.start_training_steps = start_training_steps
        
        #env info
        #set up environment
        self.env = environment(env_name)
        self.action_space = self.env.get_action_space()
        self.state_space = self.env.get_observation_space()[0]
        #print(self.state_space)
        #set up 2 network
        self.target_network = q_network(self.action_space,self.state_space)
        self.action_network = q_network(self.action_space,self.state_space)
        
        #set up game subscribe agent
        self.agent = agent(self.action_space,epsilon,self.target_network)
        
        #the storage of training experince
        self.experience_pool = Experience_pool(self.batch_size,buffer)

        #initialize optimizer
        self.optim = OPTIMIZER(params = self.target_network.parameters(), lr = 0.001)
    
    #fit the experiences to the model
    def update(self):
        self.optim.zero_grad()
        experiences = self.experience_pool.sample_experience()
        loss = computed_td_loss(self.action_network, self.target_network, experiences)
        #print(loss)
        loss.backward()
        self.optim.step()


    def training_process(self):
        reward_list = []
        time_line = []
        self.ave_reward_list = []
        sum_reward = 0
        while self.episodes < self.max_episodes:
            #s is the current step
            s = self.env.get_current_state()
            action = self.agent.take_action(s)
            reward, s_ , done = self.env.react(action)

            #'transition',['s','r','a','target','max_s_a','done']
            self.experience_pool.store_experience([s,reward,action,s_,done])
            self.total_steps += 1
            if reward != -20:
                sum_reward += reward
            if sum_reward >= 201:
                print('ok')
            
            
            #save model every 1000 steps
            #if self.total_steps > self.start_training_steps+1000 and self.total_steps % 1000 == 0:
                #print('save model!!!')
                #self.target_network.model.save_weights("%s.h5"%self.agent_name)

            #start training
            if self.total_steps > self.start_training_steps and self.total_steps % 5 == 0:
               self.update()
               if self.agent.epsilon > 0.001:
                  self.agent.epsilon *= 0.995
            
            if self.total_steps > self.start_training_steps and self.total_steps % 10 == 0:
                  #print("update behavior network")
                  self.action_network.load_state_dict(self.target_network.state_dict())
            #since the reward of fail is -10
            if self.env.isepisode_done() or sum_reward >= 201:
               reward_list.append(sum_reward)
               time_line.append(self.total_steps)
               #print("the reward at %d is %d"%(self.total_steps, sum_reward+10))
               sum_reward = 0
               self.episodes += 1
               #get average score of every 100 episodes
               if self.episodes % 10 == 0:
                   ave_reward = np.mean(reward_list)
                   self.ave_reward_list.append(ave_reward)
                   reward_list = []
                   print('Episode {} Average Reward: {}'.format(self.episodes, ave_reward))
               self.env.reset()
        return self.ave_reward_list


def get_stat(gamma):
    test_times = 2
    stats = np.zeros(max_episodes//10)
    for j in range(test_times):
                game = dqn_agent(gamma)
                stats += np.array(game.training_process())
    stats /= test_times
    return stats

def plot_graph_DQN():
        plot_list = []
        test_times = 2
        #test model with different gamma
        for i in [0.995, 0.99, 0.95, 0.9, 0.85]:
            global gamma
            gamma = i
            stats = np.zeros(max_episodes//10)
            for j in range(test_times):
                game = dqn_agent()
                stats += np.array(game.training_process())
            stats /= test_times
            plt.plot(10 * (np.arange(len(stats)) + 1), stats, label = "gamma = %.3f"%i)
        plt.legend(loc='upper left')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('DQN Average Reward vs Episodes')
        plt.savefig('%s_rewards.jpg'%"DQN")
        plt.close()
          
def save_data(ave_reward_lists):
    real_ave_reward_list = []
    for i in range(0, len(ave_reward_lists[0])):
      real_ave_reward_list.append( (ave_reward_lists[0][i] + ave_reward_lists[1][i] + ave_reward_lists[2][i] + ave_reward_lists[3][i]) / 4)
    file = open('dqn.csv', 'w+', newline ='')
    with file:    
      write = csv.writer(file)
      write.writerows(map(lambda x: [x], real_ave_reward_list))    



class DDQN_agent(dqn_agent):
    def __init__(self):
        super().__init__()
        self.agent_name = "DDQN"
        #set up 2 network
        #self.target_network = q_network(self.action_space,self.state_space)
        #self.action_network = q_network(self.action_space,self.state_space,self)
        #self.agent = agent(self.action_space,epsilon,self.action_network)
    
    #rewrite the update function
    #def update(self):
        #experiences = self.experience_pool.sample_experience()
        #self.action_network.train(experiences,self.action_network)
        #if self.total_steps % 3:
           #self.soft_update()
    

    #update the action brain
    def soft_update(self):
        self.action_network.load_state_dict(self.target_network.state_dict())

if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool_list = []
    ave_reward_lists = []
    for gamma in [0.995, 0.99, 0.95, 0.9, 0.85]:
        pool_list.append(pool.apply_async(get_stat, (gamma, )))
    #get result from the pool
    ave_reward_lists = [stat.get() for stat in pool_list]
    pool.close()
    pool.join()
    print(ave_reward_lists)
    #plot_graph_DQN()
    #for i in range(4):
       #game = DDQN_agent()
        #ave_reward_lists.append(game.training_process())
    #save_data(ave_reward_lists)
        
