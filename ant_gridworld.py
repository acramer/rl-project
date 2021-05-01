from random import random, randint
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

ACTIONS = [ (-1, 0), # North
            (-1, 1),
            ( 0, 1), # East
            ( 1, 1),
            ( 1, 0), # South
            ( 1,-1),
            ( 0,-1), # West
            (-1,-1), ]

plt.ion()

class AntGridworld:
    class State:
        def __init__(self, env_size):
            self.grid_space = np.zeros((env_size, env_size))
            self.food_space = np.zeros((env_size, env_size))
            self.trail_space = np.zeros((env_size, env_size))
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

    def __init__(self,ant_agent,env_size=20, food_num=15, num_ants=10, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10, memory_len=20):
        self.ant_agent   = ant_agent
        self.env_size    = env_size
        self.food_num    = food_num
        self.num_ants    = num_ants   
        self.max_wt      = max_wt
        self.nest_loc    = nest_loc
        self.nest_range  = nest_range
        self.obstacle_no = obstacle_no
        self.memory_len  = memory_len
        self.actions = ACTIONS
        self.reset()  

    def reset(self, env_size=None, food_num=None, num_ants=None, max_wt=None, nest_loc=None, nest_range=None, obstacle_no=None, memory_len=None, ant_agent=None):
        if env_size    is None : env_size    = self.env_size   
        if food_num    is None : food_num    = self.food_num   
        if num_ants    is None : num_ants    = self.num_ants   
        if max_wt      is None : max_wt      = self.max_wt     
        if nest_loc    is None : nest_loc    = self.nest_loc   
        if nest_range  is None : nest_range  = self.nest_range 
        if obstacle_no is None : obstacle_no = self.obstacle_no
        if memory_len  is None : memory_len  = self.memory_len
        if ant_agent   is None : ant_agent   = self.ant_agent

        self.state = AntGridworld.State(env_size)
        self.done = False
        self.ants          = []
        self.expl_ants     = []
        self.has_food      = []
        self.action_mem    = []
        self.last_states   = []
        self.ant_locations = []

        self.totalFoodCollected = 0

        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))

        self.state_mem  = np.concatenate((np.ones((num_ants,memory_len)).reshape(num_ants,memory_len,1)*self.nest[0],np.ones((num_ants,memory_len)).reshape(num_ants,memory_len,1)*self.nest[0]),2)

        self.init_food_obstacles(food_num, max_wt, nest_range,obstacle_no)
        for idx in range(num_ants):
            self.addAntToColony(idx)

        self.antIndex = 0
        self.total_starting_food = self.state.remaining_food()

        return self.get_state()

    def storeFood(self):
        self.totalFoodCollected += 1

    def init_food_obstacles(self, food_num, max_wt, nest_range, obstacle_no):
        foods = np.random.choice(np.arange(1,max_wt+1),food_num,replace=True)
        # Area surrounding the nest with range nest_range
        nest_area =  [(i,j) for j in range(self.nest[1]-nest_range, self.nest[1]+nest_range+1) \
            if  j >= 0 and j < self.state.grid_space.shape[1] \
                for i in range(self.nest[0]-nest_range, self.nest[0]+nest_range+1) \
                    if  i >= 0 and i < self.state.grid_space.shape[0]]
        # All the other points except nest_area
        cand_points =  [(i,j) for i in np.arange(self.state.grid_space.shape[0]) for j in np.arange(self.state.grid_space.shape[1]) \
             if (i,j) not in nest_area]
        
        obstacle_pts = np.random.choice(len(cand_points), obstacle_no, replace=False)
        for i in range(obstacle_no):
            obstacle_length = np.random.randint(2, 4)
            location = cand_points[obstacle_pts[i]]
            if np.random.rand()>0.5:
                locations = [(location[0],location[1]+n) for n in range(obstacle_length)]
            else:
                locations = [(location[0]+n,location[1]) for n in range(obstacle_length)]
            self.state.add_obstacle(locations)

        cand_points = [i for i in cand_points if i not in self.state.total_obs_pts]
        idx = np.random.choice(len(cand_points),food_num,replace=False)
        food_locs = [cand_points[i] for i in idx]
        self.food_locs = food_locs
        self.state.add_food(food_locs, foods)

    def addAntToColony(self, idx):
        self.ants.append(self.ant_agent(idx, self, is_exploring))
        self.has_food.append(0)
        self.action_mem.append(0)
        self.last_states.append(self.nest)
        self.ant_locations.append(self.nest)

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        obstacles  = list(np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        food  = list(np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())

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

    # def get_reward(self, state, action):
    def get_reward(self, action):
        """ TODO
        as discussed, assign various rewards for various state,action pairs.
        """
        # state = self.get_state()
        # ant = self.antIndex
         ###########################################
        # r, c = ant.get()      # get current location
        # ant_loc = (r+action[0], c+action[1])      # get new location
        # food_val, trail_val, obstacle, is_explored = self.state.food_space[ant_loc],/
        # self.state.trail_space[ant_loc],self.state.grid_space[ant_loc], self.state.explored_space[ant_loc]

        # total_reward = 0
        # if obstacle>0 :
        #     total_reward += -1
        # else:
        #     ant.set(action)
        #     if ant.is_exploring:
        #         if ant.is_foraging:
        #             if is_explored==0:  
        #                 total_reward += 1
        #                 self.state.explored_space[ant_loc] = 1
        #             if food_val>0:
        #                 total_reward += 10
        #                 ant.is_foraging = False
        #                 self.state.food_space[ant_loc] -= 1
        #             elif trail_val>0:   total_reward += 0
        #         else:
        #             if ant_loc == self.nest:
        #                 total_reward += 75
        #                 ant.is_foraging = True
        #                 self.storeFood()
        #             else:
        #                 total_reward += -1
        #                 self.state.trail_space[ant_loc] += 1
        #     else:
        #         if ant.is_foraging:
        #             if is_explored==0:  
        #                 total_reward += 0
        #                 self.state.explored_space[ant_loc] = 1
        #             if food_val>0:
        #                 total_reward += 5
        #                 ant.is_foraging = False
        #                 self.state.food_space[ant_loc] -= 1
        #             elif trail_val>0:   total_reward += 1
        #         else:
        #             if ant_loc == self.nest:
        #                 total_reward += 100
        #                 ant.is_foraging = True
        #                 self.storeFood()
        #             elif trail_val>0:
        #                 total_reward += -1
        #                 self.state.trail_space[ant_loc] += 1
        #             else:
        #                 total_reward += -2
        # if np.sum(self.state.food_space) == 0:
        #     self.done = True
        # return ant_loc, total_reward, self.done
        ######################################
        return 0

    def step(self, action):
        antID = self.antIndex
        reward = self.get_reward(action)

        self.action_mem[antID] = action + (8 if self.has_food[antID] else 0)

        # Calculating Next States
        r,c = self.ant_locations[antID]
        dr,dc = self.actions[action]
        nr = min(max(r+dr,0),self.state.grid_space.shape[0]-1)
        nc = min(max(c+dc,0),self.state.grid_space.shape[0]-1)
        if self.state.grid_space[nr,nc]:
            nr,nc = r,c

        self.ant_locations[antID] = (nr,nc)

        if self.has_food[antID]:
            if (nr,nc) == self.nest:
                self.storeFood()
                self.has_food[antID] = 0
            else:
                self.state.trail_space[nr,nc] += 1
        elif self.state.food_space[nr,nc]:
            self.state.food_space[nr,nc] -= 1
            self.state.food_space[self.state.food_space<0] = 0
            self.has_food[antID] = 1

        next_state = self.get_state()

        self.last_states[antID] = (nr,nc)
        if self.antIndex >= len(self.ants) - 1:
            self.state_mem = np.concatenate((self.state_mem[:,1:],np.array(self.last_states).reshape(self.num_ants,1,2)),1)

        # Decrementing Trail Space and Caculating Done
        if self.antIndex >= len(self.ants) - 1:
            self.state.trail_space -= 0.05
            self.state.trail_space[self.state.trail_space<0] = 0
            self.done = self.totalFoodCollected == self.total_starting_food

        # Next Ant
        self.antIndex = (self.antIndex + 1) % len(self.ants)
        return next_state, reward, self.done

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

# class AntAgent:
class NewAntAgent:
    def __init__(self,ID,env,nest,exploring=False, mean=0, sd=1):
        self.antID = ID
        self.nest = nest
        self.location = nest
        self.exploring = exploring
        self.env = env
        self.actions = ACTIONS

    def get(self):
        return self.location

    def policy(self, state):
        return [1/len(self.actions)]*len(self.actions)

    def set(self, action):
        self.location = (self.location[0] + action[0],self.location[1] + action[1])


