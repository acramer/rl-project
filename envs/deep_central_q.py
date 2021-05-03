from envs.ant_gridworld import TensorEnvironment, AntAgent

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
import wandb 

####### CENTRALIZED DEEP Q-LEARNING #######

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

    def save_model(self,save_path, des=''):
        from os import path
        from pathlib import Path
        save_dir = path.join(path.dirname(path.abspath(__file__)), save_path, des)
        save_path = path.join(save_dir, 'model'+des+'.th')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save((self.state_dict(), self.params), save_path)
        #return torch.save((self.state_dict(), self.params), path.join(path.dirname(path.abspath(__file__)), 'cls'+des+'.th'))

    def load_model(path='cls.th'):
        from os import path
        std, params = torch.load(path.join(path.dirname(path.abspath(__file__)), path), map_location='cpu')
        r = SimpleClassifier(*params)
        r.load_state_dict(std)
        return r

class DeepCentralEnvironment(TensorEnvironment):
    def __init__(self,args,updates_interval=100,memory_len=10,**kwargs):
        super().__init__(CDQAnt,memory_len=memory_len,**kwargs)

        # State Size
        #  8 # food, #  8 # trail,
        #  8 # obstacles,
        #  2 # torch.tensor([self.has_food[antID], self.action_mem[antID]]),
        # 20 # self.state_mem[antID].flatten(),
        #  2 # to_nest,
        # 40
        self.state_size = 40

        self.args = args
        self.updates_interval = updates_interval
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.D = set()
        self.steps = 0
        if args.load_model_dir:
            self.model = SimpleClassifier.load(args.load_model_dir).to(self._device)
        else:
            self.model        = SimpleClassifier([self.state_size,self.state_size*2,self.state_size,self.state_size//2,self.num_act]).to(self._device)
            self.target_model = SimpleClassifier([self.state_size,self.state_size*2,self.state_size,self.state_size//2,self.num_act]).to(self._device)

        if args.wandb:
            if args.log_dir is not None:
                wandb.init(name=args.description, config=args, dir=args.log_dir, project="rl-project")
            else:
                wandb.init(name=args.description, config=args, project="rl-project")
            wandb.watch(self.model)

        # Loss
        if args.huber: self.loss = torch.nn.SmoothL1Loss()
        else:          self.loss = torch.nn.CrossEntropyLoss()
        # Optimizer
        if args.adam: self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        else:         self.optimizer = torch.optim.SGD( self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
        # Step Scheduler
        if args.step_schedule: self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4)
        if args.load_model_dir is None:
            self.train()
        self.model.save_model(args.save_model_dir,args.description)

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
            return -10

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
                total_reward += -5
            elif (nr,nc) == tuple(self.state_mem[antID][-2]):
                total_reward += -5
        if (nr,nc) == tuple(self.state_mem[antID][-3]):
            total_reward += -2
        if (nr,nc) == tuple(self.state_mem[antID][-4]):
            total_reward += -1
        if (nr,nc) == tuple(self.state_mem[antID][-5]):
            total_reward += -1
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

        return torch.cat((  food-obstacles,
                            trail,
                            torch.tensor([self.has_food[antID], self.action_mem[antID]]),
                            self.state_mem[antID].flatten(),
                            to_nest,
                          ))

    def train(self):
        def epsilon_pi(state):
            Q = self.model(state.unsqueeze(0))
            explore_prob = self.args.epsilon/self.num_act
            ret = [explore_prob]*self.num_act
            ret[Q.argmax()] = 1 - self.args.epsilon + explore_prob
            return ret

        for ei in range(self.args.epochs):
            state = self.soft_reset().to(self._device)
            total_rewards = 0
            total_loss = 0
            for i in range(self.args.max_steps):
                action = np.random.choice(list(range(self.num_act)),p=epsilon_pi(state))
                next_state, reward, done = self.step(action)
                next_state = next_state.to(self._device)
                total_rewards += reward
                self.D.add((state,action,reward, None if done else next_state))

                sample = list(self.D)
                if len(self.D) > self.args.batch_size:
                    sample = Sample(list(self.D), self.args.batch_size)

                x,y = [],[]
                for pj,aj,rj,pnj in sample:
                    x.append(pj)
                    yj = rj if pnj is None else rj + self.args.gamma*self.target_model(pnj.unsqueeze(0)).max()
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
                if not self.steps % self.updates_interval:
                    self.target_model.load_state_dict(deepcopy(self.model.state_dict()))

                if self.args.wandb: wandb.log({"loss": loss_val}, step=self.steps)

                state = next_state
                if done: break
            
            # Step Learning Rate
            if self.args.step_schedule:
                scheduler.step(total_loss)
            if self.args.wandb:
                wandb.log({"epoch":i,"rewards":total_rewards,"food_collected":int(self.totalFoodCollected)}, step=self.steps)
            print('E:{:4d} - {:>4d}/{:4d} - Steps:{:4d} - Loss:{:8.3f} - Rewards:{:5d}'.format(ei,int(self.totalFoodCollected),int(self.total_starting_food),i,total_loss,total_rewards))
        self.soft_reset()


class CDQAnt(AntAgent):
    def __init__(self,ID,env):
        super().__init__(ID,env,'centralDQ')

    def policy(self,X):
        X.to(self.env._device)
        Q = self.env.model(X.unsqueeze(0))
        num_actions = self.env.num_act
        ret = [0]*num_actions
        ret[Q.argmax()] = 1
        return ret

