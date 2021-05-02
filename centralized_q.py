from ant_gridworld import NumpyEnvironment, TensorEnvironment, AntAgent, ACTIONS
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from random import random, randint
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


# def _build_model(self):
#     layers = self.options.layers
#     model = Sequential()

#     model.add(Dense(layers[0], input_dim=self.state_size, activation='relu'))
#     if len(layers) > 1:
#         for l in layers[1:]:
#             model.add(Dense(l, activation='relu'))
#     model.add(Dense(self.action_size, activation='linear'))
#     model.compile(loss=huber_loss,
#                   optimizer=Adam(lr=self.options.alpha))
#     return model

class SimpleClassifier(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.params = [layer_sizes]

        assert len(layer_sizes) >= 2

        temp_layer_sizes = layer_sizes[:]
        layers = []

        in_layer = temp_layer_sizes.pop(0)
        for out_layer in temp_layer_sizes:
            layers.append(nn.Linear(in_layer, out_layer))
            layers.append(nn.ReLU())
            in_layer = out_layer
        layers.pop()

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        input_size = self.params[0][0] # 28 ** 2
        output_size = self.params[0][-1] # 28 ** 2
        return self.net(x.view(-1,input_size)).view(-1,output_size)

    def classify(self, x):
        return self.forward(x).argmax(dim=1)

    def save_model(self, des=''):
        from os import path
        return torch.save((self.state_dict(), self.params), path.join(path.dirname(path.abspath(__file__)), 'cls'+des+'.th'))

    def load_model():
        from os import path
        std, params = torch.load(path.join(path.dirname(path.abspath(__file__)), 'cls.th'), map_location='cpu')
        r = SimpleClassifier(*params)
        r.load_state_dict(std)
        return r

class DecentralizedEnvironment(NumpyEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,**kwargs):
        super().__init__(DQAnt,**kwargs)
        self.num_act = 8 
        self.Q = defaultdict(lambda: np.zeros(self.num_act))
        self.rewards = []
        # self.train(epochs,max_steps,epsilon)

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
                self.state.trail_space = self.state.trail_space-0.01
                self.state.trail_space[self.state.trail_space<0] = 0
            self.rewards.append(total_reward)

class DQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralQ')
        self.state = self.env.nest
        self.rewards = []
        self.action_memory = 0

    # def policy(self,state):
    #     # food, trail, obstacles, has_food, action_memory, state_memory, location, to_nest = X[:8],X[8:16],X[16:24],X[24],X[25],X[26:66],tuple(X[66:68]),X[68:]
    #     ret = [epsilon/len(self.Q[state])]*len(self.Q[state])
    #     ret[np.argmax(self.Q[state])] = 1 - epsilon + epsilon/len(self.Q[state])
    #     return ret


class DeepCentralEnvironment(TensorEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,load=False,**kwargs):
        super().__init__(CDQAnt,**kwargs)
        self.num_act = 8 
        self.state_size = 1#WHAT

        self.D = set()
        self.steps = 0

        # TODO: init huber loss
        # TODO: init optimizer
        if load:
            self.model = SimpleClassifier.load()
        else
            self.model        = SimpleClassifier()
            self.target_model = SimpleClassifier()
            self.train(epochs,max_steps,epsilon)

    def train(self,epochs=1,max_steps=600,epsilon=0.4,alpha=0.1,gamma=0.1):
        def epsilon_pi(observation):
            # TODO
        for ei in range(epochs):
            state = tuple(self.reset())
            for i in range(max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                # next_state = tuple(next_state)
                # self.D.add((tuple(state),action,reward, None if done else tuple(new_state)))
                self.D.add(state,action,reward, None if done else new_state))

                sample = list(self.D)
                if len(self.D) > self.options.replay_memory_size:
                    sample = random.sample(list(self.D),self.options.replay_memory_size)

                x,y = [],[]
                for pj,aj,rj,pnj in sample:
                    # x.append(list(pj))
                    x.append(pj)
                    # pj, pnj = np.array(pj), np.array(pnj)
                    yj = rj if pnj is None else rj + gamma*self.target_model(pnj.unsqueeze(0)).max()
                    yt = self.model(pj.unsqueeze(0))[0].copy()
                    yt[aj] = yj
                    y.append(yt)

                # x = np.array(x).unsqueeze(0)
                x = torch.cat(x,0)
                y = torch.cat(y,0)
                # self.model.fit(x,y, batch_size=self.options.batch_size, verbose=0)
                # TODO: grad zero
                # TODO: huber loss
                # TODO: backward
                # TODO: op.step

                self.steps += 1
                if not self.steps % C:
                    self.target_model.load_state_dict(deepcopy(self.model.state_dict()))

                state = new_state
                if done: break
            print('{:4d} - {:4d}/{:4d} - {:4d}'.format(ei,int(self.totalFoodCollected),int(self.total_starting_food),i))

class CDQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralDQ')

    def policy(self,X):
        # TODO
        Q = env.model(state.unsqueeze(0))
        num_actions = env.num_act
        ret = [0]*num_actions
        ret[Q.argmax()] = 1
        return ret


# import os
# import random
# from collections import deque
# import tensorflow as tf
# from keras import backend as bk
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.losses import huber_loss
# import numpy as np
# from Solvers.Abstract_Solver import AbstractSolver
# from lib import plotting
# 
# 
#     def make_epsilon_greedy_policy(self):
#         nA = self.env.action_space.n
# 
#         def policy_fn(state):
#             Q = self.model.predict(state.reshape([1,-1]))
#             num_actions = self.action_size
#             explore_prob = self.options.epsilon/num_actions
# 
#             ret = [explore_prob]*num_actions
#             ret[np.argmax(Q,axis=1)[0]] = 1 - self.options.epsilon + explore_prob
#             return ret
# 
# 
#         return policy_fn
# 
#     def train_episode(self):
#         state = self.env.reset()
#         C = self.options.update_target_estimator_every
# 
#         pi = self.make_epsilon_greedy_policy()
#         done = False
# 
#         while not done:
#             choice_dist = pi(state)
#             action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
#             new_state, reward, done, _ = self.step(action)
#             self.D.add((tuple(state),action,reward, None if done else tuple(new_state)))
# 
#             sample = list(self.D)
#             if len(self.D) > self.options.replay_memory_size:
#                 sample = random.sample(list(self.D),self.options.replay_memory_size)
# 
#             x = []
#             y = []
# 
#             for pj,aj,rj,pnj in sample:
#                 x.append(list(pj))
#                 pj, pnj = np.array(pj), np.array(pnj)
#                 yj = rj if np.any(pnj == np.array(None)) else rj + np.max(self.options.gamma*self.target_model.predict(pnj.reshape([1,-1])))
#                 yt = self.model.predict(pj.reshape([1,-1]))[0].copy()
#                 yt[aj] = yj
#                 y.append(yt)
# 
#             x = np.array(x).reshape([-1, self.state_size])  # unsqueeze(0)
#             y = np.array(y).reshape([-1, self.action_size]) # unsqueeze(0)
#             self.model.fit(x,y, batch_size=self.options.batch_size, verbose=0) # grad zero, loss, backward, op.step
# 
#             self.steps += 1
#             if not self.steps % C:
#                 self.update_target_model()
# 
#             state = new_state
# 
#     def create_greedy_policy(self):
#         nA = self.env.action_space.n
# 
#         def policy_fn(state):
# 
#         return policy_fn
