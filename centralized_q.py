from ant_gridworld import NumpyEnvironment, TensorEnvironment, AntAgent, ACTIONS
import torch
import torch.nn as nn
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
        super().__init__(ID,env,'decentralQ')
        self.state = self.env.nest
        self.rewards = []
        self.action_memory = 0

    # def policy(self,state):
    #     # food, trail, obstacles, has_food, action_memory, state_memory, location, to_nest = X[:8],X[8:16],X[16:24],X[24],X[25],X[26:66],tuple(X[66:68]),X[68:]
    #     ret = [epsilon/len(self.Q[state])]*len(self.Q[state])
    #     ret[np.argmax(self.Q[state])] = 1 - epsilon + epsilon/len(self.Q[state])
    #     return ret


####### CENTRALIZED DEEP Q-LEARNING #######

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

        # Complete this with slices
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

class DeepCentralEnvironment(TensorEnvironment):
    def __init__(self,epochs=1,max_steps=600,epsilon=0.4,load=False,replay_memory_size=100,**kwargs):
        super().__init__(CDQAnt,**kwargs)
        self.replay_memory_size = replay_memory_size

        self.D = set()
        self.steps = 0
        if load:
            self.model = SimpleClassifier.load()
        else:
            self.model        = SimpleClassifier([self.state_size,self.state_size//2,self.num_act])
            self.target_model = SimpleClassifier([self.state_size,self.state_size//2,self.num_act])

        # TODO: self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # TODO: learning args
        # Loss
        # loss = torch.nn.CrossEntropyLoss()
        # self.loss = torch.nn.HuberLoss()
        self.loss = torch.nn.SmoothL1Loss()
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        # if args.adam: optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        # else:         optimizer = torch.optim.SGD( model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
        # Step Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4)
        # if args.step_schedule: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
        if not load:
            self.train(epochs,max_steps,epsilon)

    def train(self,epochs=1,max_steps=600,epsilon=0.4,alpha=0.1,gamma=0.1,updates_interval=10):
        def epsilon_pi(state):
            Q = self.model(state.unsqueeze(0))
            explore_prob = epsilon/self.num_act
            ret = [explore_prob]*self.num_act
            ret[Q.argmax()] = 1 - epsilon + explore_prob
            return ret
        for ei in range(epochs):
            state = self.reset()
            total_rewards = 0
            total_loss = 0
            for i in range(max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                total_rewards += reward
                # next_state = tuple(next_state)
                # self.D.add((tuple(state),action,reward, None if done else tuple(new_state)))
                self.D.add((state,action,reward, None if done else next_state))

                sample = list(self.D)
                if len(self.D) > self.replay_memory_size:
                    sample = Sample(list(self.D),self.replay_memory_size)

                x,y = [],[]
                for pj,aj,rj,pnj in sample:
                    # x.append(list(pj))
                    x.append(pj)
                    # pj, pnj = np.array(pj), np.array(pnj)
                    yj = rj if pnj is None else rj + gamma*self.target_model(pnj.unsqueeze(0)).max()
                    yt = self.model(pj.unsqueeze(0)).clone()
                    yt[0,aj] = yj
                    y.append(yt)

                # x = np.array(x).unsqueeze(0)
                x = torch.cat(x,0)
                y = torch.cat(y,0)

                # Loss
                loss_val = self.loss(self.model(x), y)
                # Grad Step
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()
                total_loss += loss_val.item()

                self.steps += 1
                if not self.steps % updates_interval:
                    self.target_model.load_state_dict(deepcopy(self.model.state_dict()))

                state = next_state
                if done: break
            
            # Step Learning Rate
            # if args.step_schedule:
            #     scheduler.step(total_loss)
            self.scheduler.step(total_loss)
            print('E:{:4d} - {:>4d}/{:4d} - Steps:{:4d} - Rewards:{:5d}'.format(ei,int(self.totalFoodCollected),int(self.total_starting_food),i,total_rewards))

class CDQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralDQ')

    def policy(self,X):
        Q = self.env.model(X.unsqueeze(0))
        num_actions = self.env.num_act
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
