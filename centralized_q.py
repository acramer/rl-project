from ant_gridworld import AntGridworld, AntAgent, ACTIONS
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from random import random, randint
from time import sleep
from collections import defaultdict

class NumpyEnvironment(AntGridworld):
    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        actView=[1,2,5,8,7,6,3,0]
        obstacles = np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        trail     = np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = np.zeros(2)
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return np.concatenate((food,
                               # trail,
                               # obstacles,
                               [self.has_food[antID]],
                               # self.action_mem[antID]],
                               # self.state_mem[antID].flatten(),
                               # self.ant_locations[antID],
                               to_nest,
                             ))

class TensorEnvironment(AntGridworld):
    class State:
        def __init__(self, env_size):
            self.grid_space     = torch.zeros((env_size, env_size))
            self.food_space     = torch.zeros((env_size, env_size))
            self.trail_space    = torch.zeros((env_size, env_size))
            self.explored_space = torch.zeros((env_size, env_size))
            self.total_obs_pts = set()
            self.env_size = env_size

        def add_food(self, food_locs, foods):
            for i in range(len(foods)):
                self.food_space[food_locs[i]] += foods[i]

        def add_trail(self, location, weight):
            self.trail_space[location] = weight

        def add_obstacle(self, locations):
            for loc in locations:
                if loc[0]<self.env_size and loc[1]<self.env_size:
                    self.grid_space[loc] = 1
                    self.total_obs_pts.add(loc)

        def remaining_food(self):
            return np.sum(self.food_space)

    # TODO: learn embedding

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        actView=[1,2,5,8,7,6,3,0]
        obstacles = np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        trail     = np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = np.zeros(2)
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return np.concatenate((food,
                               trail,
                               obstacles,
                               [self.has_food[antID],
                               self.action_mem[antID]],
                               self.state_mem[antID].flatten(),
                               self.ant_locations[antID],
                               to_nest,
                             ))



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
        for _ in range(epochs):
            done = False
            state = tuple(self.reset())
            for i in range(max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                next_state = tuple(next_state)
                self.Q[state][action] = self.Q[state][action] + alpha*(reward+gamma*max(self.Q[next_state])- self.Q[state][action])
                state = next_state
                if done: break
            print('Last Step:',i)
        self.reset()

class CQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralQ')

    def policy(self,X):
        # food, trail, obstacles, has_food, action_memory, state_memory, location, to_nest = X[:8],X[8:16],X[16:24],X[24],X[25],X[26:66],tuple(X[66:68]),X[68:]
        ret = [0]*self.env.num_act
        ret[np.argmax(self.env.Q[tuple(X)])] = 1
        return ret


class DeepCentralEnvironment(TensorEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(CDQAnt,**kwargs)
        self.num_act = 8 
        self.Q = defaultdict(lambda: torch.zeros(self.num_act))
        self.train(epochs,max_steps,epsilon)

    def train(self,epochs=1,max_steps=600,epsilon=0.4,alpha=0.1,gamma=0.1):
        def epsilon_pi(observation):
            observation = tuple(observation)
            ret = [epsilon/len(self.Q[observation])]*len(self.Q[observation])
            ret[np.argmax(self.Q[observation])] = 1 - epsilon + epsilon/len(self.Q[observation])
            return ret
        for _ in range(epochs):
            done = False
            state = tuple(self.reset())
            for i in range(max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                next_state = tuple(next_state)
                self.Q[state][action] = self.Q[state][action] + alpha*(reward+gamma*max(self.Q[next_state])- self.Q[state][action])
                state = next_state
                if done: break
            print('Last Step:',i)


class CDQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralDQ')

    def policy(self,X):
        ret = [0]*self.env.num_act
        ret[np.argmax(self.env.Q[tuple(X)])] = 1
        return ret


