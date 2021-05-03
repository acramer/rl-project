from ant_gridworld import NumpyEnvironment, TensorEnvironment, AntAgent, ACTIONS

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
        self.rewards = []
        self.left_food = []

    # TODO: get hyper params epsilon,alpha,gamma
    def train(self,epochs=1,max_steps=600,epsilon=0.7,alpha=0.5,gamma=0.9):
        def epsilon_pi(observation):
            ret = [epsilon/len(self.Q[observation])]*len(self.Q[observation])
            ret[np.argmax(self.Q[observation])] = 1 - epsilon + epsilon/len(self.Q[observation])
            return ret
        for _ in range(epochs):
            total_reward = 0
            for i in np.arange(self.num_ants):
                done = False
                state = (self.ant_locations[i],self.has_food[i])
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state)) 
                _ , reward, done = self.step(action)
                total_reward += reward
                next_state = (self.ant_locations[i],self.has_food[i])
                self.Q[state][action] = self.Q[state][action] + alpha*(reward+gamma*max(self.Q[next_state])- self.Q[state][action])
                    # state = next_state
                if done: break
            self.state.trail_space = self.state.trail_space-0.05
            self.state.trail_space[self.state.trail_space<0] = 0
            self.rewards.append(total_reward/self.num_ants)
            self.left_food.append(self.state.remaining_food())

class JQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'jointQ')


class DecentralizedEnvironment(NumpyEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(DecQAnt,**kwargs)
        self.num_act = 8 
        self.rewards = []
        self.left_food = []

    # TODO: get hyper params epsilon,alpha,gamma
    def train(self,epochs=1,max_steps=600,epsilon=0.7,alpha=0.5,gamma=0.9):
        def epsilon_pi(ant,observation):
            ret = [epsilon/len(ant.Q[observation])]*len(ant.Q[observation])
            ret[np.argmax(ant.Q[observation])] = 1 - epsilon + epsilon/len(ant.Q[observation])
            return ret
        for _ in range(epochs):
            total_reward = 0
            for idx, ant in enumerate(self.ants):
                done = False
                state = (self.ant_locations[idx],self.has_food[idx])
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(ant,state)) 
                _ , reward, done = self.step(action)
                total_reward += reward
                next_state = (self.ant_locations[idx],self.has_food[idx])
                ant.Q[state][action] = ant.Q[state][action] + alpha*(reward+gamma*max(ant.Q[next_state]) - ant.Q[state][action])
                    # state = next_state
                if done: break
            self.state.trail_space = self.state.trail_space-0.05
            self.state.trail_space[self.state.trail_space<0] = 0
            self.rewards.append(total_reward/self.num_ants)
            self.left_food.append(self.state.remaining_food())

class DecQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'decQ')
        self.Q = defaultdict(lambda: np.zeros(self.env.num_act))



