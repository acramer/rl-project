from envs.ant_gridworld import AntGridworld, AntAgent, ACTIONS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from random import random, randint
from time import sleep

class ProceduralEnvironment(AntGridworld):
    # def __init__(self,env_size=20, food_num=15, num_ants=10, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10, memory_len=20):
    def __init__(self,**kwargs):
        super().__init__(ProceduralAnt,**kwargs)

    def addAntToColony(self, idx):
        is_exploring = np.random.rand() <= 0.15

        if is_exploring:
            ant = self.ant_agent(idx, self, 'exploring')
            self.expl_ants.append(ant)
        else:
            ant = self.ant_agent(idx, self, 'exploiting')

        self.ants.append(ant)
        self.has_food.append(0)
        self.action_mem.append(0)
        self.ant_locations.append(self.nest)


class ProceduralAnt(AntAgent):
    def __init__(self,ID,env,ant_type='exploiting',memory_size=20,mean=0,sd=0.5):
        assert ant_type in {'exploiting','exploring'}, "ant_type of ProceduralAnt must be in"+str({'exploiting','exploring'})
        super().__init__(ID,env,ant_type)
        rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
        normal_dist = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2)
        self.nd = list(normal_dist/np.sum(normal_dist))

    def policy(self,X):
        food, trail, obstacles, has_food, action_memory, state_memory, location, to_nest = X[:8],X[8:16],X[16:24],X[24],X[25],X[26:66],tuple(X[66:68]),X[68:]
        state_memory = [(int(x),int(y)) for x,y in zip(state_memory[::2],state_memory[1::2])]
        to_nest = [tuple(to_nest),(0,to_nest[1]),(to_nest[0],0)]
        if action_memory > 7:
            action_memory = action_memory % 8 if has_food else randint(0,7)
        if has_food:
            for v,a in enumerate(self.actions):
                if obstacles[v]:
                    trail[v] = -1
                elif a in to_nest and (location[0]+a[0],location[1]+a[1]) not in state_memory[-3:]:
                    trail[v] += 0.5
                elif (location[0]+a[0],location[1]+a[1]) in state_memory:
                    trail[v] = -1*random()
                else:
                    trail[v] = -0.5*random()
            act = np.argmax(trail)
        else:
            act = self.explore(food,trail,action_memory,state_memory,location) if self.type == 'exploring' else self.forage(food,trail,action_memory,state_memory,location)
        ret = [0]*8
        ret[act] = 1
        return ret

    def explore(self,food,trail,action_memory,state_memory,location):
        if max(food) > 0:
            act = np.argmax(food)
        elif max(trail) > 0:
            dist = max(abs(location[0]-self.env.nest[0]),abs(location[1]-self.env.nest[1]))
            v = [idx for idx,val in enumerate(trail) if val!=0]
            for i in np.arange(len(v)):
                a = self.actions[v[i]]
                next_step = (location[0]+a[0],location[1]+a[1])
                dist2 = max(abs(next_step[0]-self.env.nest[0]),abs(next_step[1]-self.env.nest[1]))
                if dist2 <= dist or next_step in state_memory:
                    trail[v[i]] = 0
            act = np.argmax(trail) if max(trail) > 0 else self.walk(food,action_memory)
        else:
            act = self.walk(food,action_memory)
        return act

    def forage(self,food,trail,action_memory,state_memory,location):
        if max(food) > 0: 
            act = np.argmax(food)
        elif max(trail) > 0:
            dist = max(abs(location[0]-self.env.nest[0]),abs(location[1]-self.env.nest[1]))
            v = [idx for idx,val in enumerate(trail) if val>0]
            for i in np.arange(len(v)):
                a = self.actions[v[i]]
                next_step = (location[0]+a[0],location[1]+a[1])
                dist2 = max(abs(next_step[0]-self.env.nest[0]),abs(next_step[1]-self.env.nest[1]))
                if dist2 <= dist :
                    trail[v[i]] = 0
            act = np.argmax(trail) if max(trail) > 0 else self.walk(food,action_memory)
        else:
            act = self.walk(food,action_memory)
        return act

    def walk(self,food,action_mem):
        pdist = self.nd[-1*action_mem:] + self.nd[:-1*action_mem]
        step_idx = [idx for idx,val in enumerate(food) if val==-1]
        if any(step_idx):
            for i in step_idx:
                pdist[i] = 0
            pdist = pdist/sum(pdist)
        return np.random.choice(8,p=pdist)

