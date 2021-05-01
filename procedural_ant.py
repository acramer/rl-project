from ant_gridworld import AntGridworld, ACTIONS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from random import random, randint
from time import sleep

class ProceduralEnvironment(AntGridworld):
    def __init__(self,env_size=20, food_num=15, num_ants=10, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10, memory_len=20):
        super().__init__(ProceduralAnt,env_size,food_num,num_ants,max_wt, nest_loc,nest_range,obstacle_no,memory_len)

    def addAntToColony(self, idx):
        is_exploring = np.random.rand() <= 0.15

        ant = self.ant_agent(idx, self, is_exploring)
        if is_exploring:
            self.expl_ants.append(ant)

        self.ants.append(ant)
        self.has_food.append(0)
        self.action_mem.append(0)
        self.last_states.append(self.nest)
        self.ant_locations.append(self.nest)

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        obstacles = list(np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        food      = list(np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail     = list(np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())

        location = self.ant_locations[antID]
        to_nest = [0,0]
        if   location[0] < self.nest[0]: to_nest[0] =  1
        elif location[0] > self.nest[0]: to_nest[0] = -1
        if   location[1] < self.nest[1]: to_nest[1] =  1
        elif location[1] > self.nest[1]: to_nest[1] = -1

        return (  convert_input(food)
                + convert_input(trail)
                + convert_input(obstacles)
                + [self.has_food[antID]]
                + [self.action_mem[antID]]
                + self.state_mem[antID].flatten().tolist()
                + list(self.ant_locations[antID])
                + to_nest)

    def plot_environment(self,stepNum=-1):
        C,R = self.state.grid_space.shape
        
        # Object and Food markers 
        oY, oX = np.where(self.state.grid_space>0)
        tY, tX = np.where(self.state.trail_space>0)
        fY, fX = np.where(self.state.food_space>0)
        fS = np.array([self.state.food_space[x,y] for x,y in zip(list(fY), list(fX))])
        fS /= self.max_wt/30

        # Foraging and non foraging ants 
        antFY, antFX = [],[]
        antNFY, antNFX = [],[]
        for a in self.ants:
            ar,ac = self.ant_locations[a.antID]
            if self.has_food[a.antID]:
                antFY.append(ar)
                antFX.append(ac)
            else:
                antNFY.append(ar)
                antNFX.append(ac)

        # Plot the nest, ants, trails
        plt.clf()
        if stepNum >= 0:
            plt.annotate('Step: '+str(stepNum),(5,-1))
        plt.annotate('Food Stored: '+str(self.totalFoodCollected),(7,-1))
        plt.annotate('Food Left: '+str(self.state.remaining_food()),(11,-1))
        plt.axis([-1,C,R,-1])
        plt.scatter(self.nest[1],self.nest[0], color='#D95319', marker='s', s=70)   # Nest
        plt.scatter(tX,tY, color='#0072BD', s=4)                                    # Trail
        plt.scatter(fX, fY, color='#FF2FFF',marker='^',s=fS)
        plt.scatter(oX, oY, color='#000000',marker='s',s=70)                        # Food particles
        plt.scatter( antFX,  antFY, color='#77AC30', s=30)                          # Ants searching for food
        plt.scatter(antNFX, antNFY, color='#A2142F', s=30)                          # Ants returning with food
        plt.xticks(range(-1,C,C//10))
        plt.yticks(range(-1,R,R//10))
        plt.show()
        close_button = widgets.Button(plt.axes([0.13, 0.89, 0.2, 0.06]), "Close", hovercolor = '0.975')
        close_button.on_clicked(lambda x : exit())
        plt.pause(0.01)

class ProceduralAnt:
    def __init__(self,ID,env,memory_size=20,exploring=False,mean=0,sd=0.5):
        self.antID = ID
        self.env = env
        self.exploring = exploring
        rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
        normal_dist = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2)
        self.nd = list(normal_dist/np.sum(normal_dist))
        self.actions = ACTIONS

    def __call__(self,X):
        return self.policy(X)

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
            act = self.explore(food,trail,action_memory,state_memory,location) if self.exploring else self.forage(food,trail,action_memory,state_memory,location)
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

