from envs.ant_gridworld import NumpyEnvironment, TensorEnvironment, AntAgent, ACTIONS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from random import random, randint, sample as Sample
from time import sleep
from collections import defaultdict
from copy import deepcopy


class CentralEnvironment(NumpyEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(CQAnt,**kwargs)
        self.num_act = 8 
        self.Q = defaultdict(lambda: np.zeros(self.num_act))
        self.train(epochs,max_steps,epsilon)

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        actView=[1,2,5,8,7,6,3,0]
        obstacles = np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        # trail     = np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = np.zeros(2)
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return np.concatenate((food-obstacles,
                               # trail,
                               # obstacles,
                               [self.has_food[antID]],
                               # self.action_mem[antID]],
                               # self.state_mem[antID].flatten(),
                               # self.ant_locations[antID],
                               # to_nest,
                             ))

    # TODO: get hyper params epsilon,alpha,gamma
    def train(self,epochs=1,max_steps=600,epsilon=0.7,alpha=0.5,gamma=0.9):
        def epsilon_pi(observation):
            observation = tuple(observation)
            ret = [epsilon/len(self.Q[observation])]*len(self.Q[observation])
            ret[np.argmax(self.Q[observation])] = 1 - epsilon + epsilon/len(self.Q[observation])
            return ret
        for ei in range(epochs):
            done = False
            state = tuple(self.reset())
            for i in range(max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                next_state = tuple(next_state)
                self.Q[state][action] = self.Q[state][action] + alpha*(reward+gamma*max(self.Q[next_state])- self.Q[state][action])
                state = next_state
                if done: break
            print('{:4d} - {:4d}/{:4d} - {:4d}'.format(ei,int(self.totalFoodCollected),int(self.total_starting_food),i))
        self.reset()

class CQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralQ')

    def policy(self,X):
        # food, trail, obstacles, has_food, action_memory, state_memory, location, to_nest = X[:8],X[8:16],X[16:24],X[24],X[25],X[26:66],tuple(X[66:68]),X[68:]
        ret = [0]*self.env.num_act
        ret[np.argmax(self.env.Q[tuple(X)])] = 1
        return ret


class JointEnvironment(NumpyEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(JQAnt,**kwargs)
        self.num_act = 8 
        self.Q = defaultdict(lambda: np.zeros(self.num_act))
        self.epsilon = 0.5

    def train(self,state,action,reward,next_state,epochs=1,epsilon=0.7,alpha=0.5,gamma=0.9):
        self.Q[state][action] += alpha * (reward + gamma * max(self.Q[next_state]) - \
            self.Q[state][action])

    def get_state(self):
        idx = self.antIndex
        return (self.ant_locations[idx],self.has_food[idx])

    def step(self,action):
        idx = self.antIndex
        state = (self.ant_locations[idx], self.has_food[idx])
        _, reward, done = super().step(action)
        next_state = (self.ant_locations[idx], self.has_food[idx])
        self.train(state,action,reward,next_state)
        if done:
            self.reset()
        return next_state, reward, done

class JQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'jointQ')

    def policy(self,state):
        epsilon = self.env.epsilon
        probs = [epsilon/len(self.env.Q[state])]*len(self.env.Q[state])
        probs[np.argmax(self.env.Q[state])] = 1 - epsilon + epsilon/len(self.env.Q[state])
        return probs


class DecentralizedEnvironment(NumpyEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(DecQAnt,**kwargs)
        self.num_act = 8 
        self.epsilon = 0.5

    def train(self,idx,state,action,reward,next_state,epochs=1,epsilon=0.7,alpha=0.5,gamma=0.9):
        self.ants[idx].Q[state][action] += alpha * (reward + gamma * max(self.ants[idx].Q[next_state]) - \
            self.ants[idx].Q[state][action])

    def get_state(self):
        idx = self.antIndex
        return (self.ant_locations[idx],self.has_food[idx])

    def step(self,action):
        idx = self.antIndex
        state = (self.ant_locations[idx], self.has_food[idx])
        _, reward, done = super().step(action)
        next_state = (self.ant_locations[idx], self.has_food[idx])
        self.train(idx,state,action,reward,next_state)
        if done:
            self.reset()
        return next_state, reward, done


class DecQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'decQ')
        self.Q = defaultdict(lambda: np.zeros(self.env.num_act))
    
    def policy(self,state):
        idx = self.env.antIndex
        epsilon = self.env.epsilon
        # state = (self.env.ant_locations[idx],self.env.has_food[idx])
        probs = [epsilon/len(self.env.ants[idx].Q[state])]*len(self.env.ants[idx].Q[state])
        probs[np.argmax(self.env.ants[idx].Q[state])] = 1 - epsilon + epsilon/len(self.env.ants[idx].Q[state])
        return probs



