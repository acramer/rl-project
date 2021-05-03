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

    def load_model(path='cls.th'):
        from os import path
        std, params = torch.load(path.join(path.dirname(path.abspath(__file__)), path), map_location='cpu')
        r = SimpleClassifier(*params)
        r.load_state_dict(std)
        return r

class DeepCentralEnvironment(TensorEnvironment):
    def __init__(self,args,replay_memory_size=100,memory_len=10,**kwargs):
        super().__init__(CDQAnt,memory_len=memory_len,**kwargs)

        # State Size
        #  8 # food,
        #  8 # trail,
        #  8 # obstacles,
        #  2 # torch.tensor([self.has_food[antID], self.action_mem[antID]]),
        # 20 # self.state_mem[antID].flatten(),
        #  2 # to_nest,
        # 48
        self.state_size = 48

        self.args = args
        self.replay_memory_size = replay_memory_size
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.D = set()
        self.steps = 0
        if args.load_model_dir:
            self.model = SimpleClassifier.load(args.load_model_dir).to(self._device)
        else:
            self.model        = SimpleClassifier([self.state_size,self.state_size//2,self.num_act]).to(self._device)
            self.target_model = SimpleClassifier([self.state_size,self.state_size//2,self.num_act]).to(self._device)


        # TODO: learning args
        # Loss
        # SmoothL1 == Huber Loss
        if args.huber: self.loss = torch.nn.SmoothL1Loss()
        else:          self.loss = torch.nn.CrossEntropyLoss()
        # Optimizer
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        if args.adam: self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        else:         self.optimizer = torch.optim.SGD( self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
        # Step Scheduler
        #                      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4)
        if args.step_schedule: self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4)
        if args.load_model_dir is None:
            self.train(args.epochs,args.max_steps,args.epsilon)

    def get_reward(self, action):
        state = self.get_state()
        antID = self.antIndex
        ant = self.ants[antID]

        nr,nc = self.calc_next_location(action,avoid_obs=False)

        food_val    =     self.state.food_space[nr,nc]
        trail_val   =    self.state.trail_space[nr,nc]
        obstacle    =     self.state.grid_space[nr,nc]
        is_explored = self.state.explored_space[nr,nc]
        last_loc    = self.state_mem[antID][-1]

        total_reward = 0
        if obstacle > 0: 
            return -100

        nest_dist2 = lambda x:(x[0]-self.nest[0])**2+(x[1]-self.nest[1])**2

        rewards = [  500, 0, -1, 100, 0, -1]
        if self.has_food[antID]:
            if (nr,nc) == self.nest:
                total_reward += rewards[0]
            elif nest_dist2(last_loc) > nest_dist2((nr,nc)):
                total_reward += rewards[1]
            else:
                total_reward += rewards[2]
        else:
            if food_val>0:
                total_reward += rewards[3]
            if nest_dist2(last_loc) > nest_dist2((nr,nc)):
                total_reward += rewards[4]
            else:
                total_reward += rewards[5]
            if (nr,nc) == tuple(self.state_mem[antID][-1]):
                total_reward += -10
            elif (nr,nc) == tuple(self.state_mem[antID][-2]):
                total_reward += -10
        if (nr,nc) == tuple(self.state_mem[antID][-3]):
            total_reward += -5
        if (nr,nc) == tuple(self.state_mem[antID][-4]):
            total_reward += -3
        if (nr,nc) == tuple(self.state_mem[antID][-5]):
            total_reward += -2
        return total_reward

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        actView=[1,2,5,8,7,6,3,0]
        obstacles = F.pad(self.state.grid_space, (1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = F.pad(self.state.food_space, (1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]
        trail     = F.pad(self.state.trail_space,(1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = torch.zeros(2)
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return torch.cat((  food,
                            trail,
                            obstacles,
                            torch.tensor([self.has_food[antID], self.action_mem[antID]]),
                            self.state_mem[antID].flatten(),
                            to_nest,
                          ))

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
                self.D.add((state,action,reward, None if done else next_state))

                sample = list(self.D)
                if len(self.D) > self.replay_memory_size:
                    sample = Sample(list(self.D),self.replay_memory_size)

                x,y = [],[]
                for pj,aj,rj,pnj in sample:
                    x.append(pj)
                    yj = rj if pnj is None else rj + gamma*self.target_model(pnj.unsqueeze(0)).max()
                    yt = self.model(pj.unsqueeze(0)).clone()
                    yt[0,aj] = yj
                    y.append(yt)

                x = torch.cat(x,0).to(self._device)
                y = torch.cat(y,0).to(self._device)

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
            if self.args.step_schedule:
                scheduler.step(total_loss)
            # self.scheduler.step(total_loss)
            print('E:{:4d} - {:>4d}/{:4d} - Steps:{:4d} - Loss:{:8.3f} - Rewards:{:5d}'.format(ei,int(self.totalFoodCollected),int(self.total_starting_food),i,total_loss,total_rewards))
        self.reset()

class CDQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralDQ')

    def policy(self,X):
        Q = self.env.model(X.unsqueeze(0))
        num_actions = self.env.num_act
        ret = [0]*num_actions
        ret[Q.argmax()] = 1
        return ret

