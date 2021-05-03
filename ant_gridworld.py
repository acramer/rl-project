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
            self.grid_space     = np.zeros((env_size, env_size))
            self.food_space     = np.zeros((env_size, env_size))
            self.trail_space    = np.zeros((env_size, env_size))
            self.explored_space = np.zeros((env_size, env_size))
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

    def __init__(self,ant_agent,env_size=20, food_num=15, num_ants=10, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10, memory_len=20, **kwargs):
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
        self.ant_locations = []

        self.totalFoodCollected = 0

        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))

        self.state_mem = np.concatenate((np.ones((num_ants,memory_len)).reshape(num_ants,memory_len,1)*self.nest[0],
                                         np.ones((num_ants,memory_len)).reshape(num_ants,memory_len,1)*self.nest[1]),2)

        self.init_food_obstacles(food_num, max_wt, nest_range,obstacle_no)
        for idx in range(num_ants):
            self.addAntToColony(idx)

        self.antIndex = 0
        self.total_starting_food = self.state.remaining_food()

        return self.get_state()

    def storeFood(self):
        self.totalFoodCollected += 1

    def calc_next_location(self,action,loc=None,avoid_obs=True):
        r,c = self.ant_locations[self.antIndex] if loc is None else loc
        dr,dc = self.actions[action]
        nr = min(max(r+dr,0),self.state.grid_space.shape[0]-1)
        nc = min(max(c+dc,0),self.state.grid_space.shape[0]-1)
        if avoid_obs and self.state.grid_space[nr,nc]:
            nr,nc = r,c
        return nr,nc

    def init_food_obstacles(self, food_num, max_wt, nest_range, obstacle_no):
        foods = np.random.choice(np.arange(1,max_wt+1),food_num,replace=True)
        # Area surrounding the nest with range nest_range
        rows,cols = self.state.grid_space.shape[0], self.state.grid_space.shape[1]
        nx,ny = self.nest
        nest_area =  [(i,j) for j in range(ny-nest_range, ny+nest_range+1) \
            if  j >= 0 and j < cols \
                for i in range(nx-nest_range, nx+nest_range+1) \
                    if  i >= 0 and i < rows]
        # All the other points except nest_area
        cand_points =  [(i,j) for i in range(rows) for j in range(cols) if (i,j) not in nest_area]
        
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
        self.ants.append(self.ant_agent(idx, self))
        self.has_food.append(0)
        self.action_mem.append(0)
        self.ant_locations.append(self.nest)

    def get_state(self, antID=None):
        antID = self.antIndex if antID is None else antID
        r, c  = self.ant_locations[antID]
        actView = [1,2,5,8,7,6,3,0]
        obstacles = np.pad(self.state.grid_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]
        trail     = np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = [0,0]
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return (  list(food)
                + list(trail)
                + list(obstacles)
                + [self.has_food[antID]]
                + [self.action_mem[antID]]
                + list(self.state_mem[antID].flatten())
                + list(self.ant_locations[antID])
                + to_nest)

    def get_reward(self, action):
        state = self.get_state()
        antID = self.antIndex
        ant = self.ants[antID]

        # Calculating Next Location
        nr,nc = self.calc_next_location(action,avoid_obs=False)

        food_val    =     self.state.food_space[nr,nc]
        trail_val   =    self.state.trail_space[nr,nc]
        obstacle    =     self.state.grid_space[nr,nc]
        is_explored = self.state.explored_space[nr,nc]
        last_loc    = self.state_mem[antID][-1]

        total_reward = 0
        if obstacle > 0: 
            return -1 # total_reward += -1

        # rewards: dropping food at nest [0]
        #          returning and using trail [1]
        #          returning and not using trail [2]
        #          reducing distance from nest while returning [3]
        #          finding new gridpoint while foraging [4]
        #          finding food [5]
        #          walking on trail while foraging [6]
        #          increasing distance from nest while foraging [7]

        exploring_rewards  = [  20, -1, 1, 1, 1, 10,  0, -1]
        exploiting_rewards = [  20, -1, 1, 1, 1, 10,  0, -1]

        # Squared distance to nest
        nest_dist2 = lambda x:(x[0]-self.nest[0])**2+(x[1]-self.nest[1])**2

        rewards = exploiting_rewards
        if ant.type == 'exploring' :
            rewards = exploring_rewards
        if self.has_food[antID]: # not ant.is_foraging
            if (nr,nc) == self.nest:
                total_reward += rewards[0]
            elif trail_val>0:
                total_reward += rewards[1]
            else:
                total_reward += rewards[2]
            if nest_dist2(last_loc) > nest_dist2((nr,nc)):
                total_reward += rewards[3]
            # else:
                # total_reward += -rewards[3]
        else:
            if not is_explored:  
                total_reward += rewards[4]
            if food_val>0:
                total_reward += rewards[5]
            elif trail_val>0:
                total_reward += rewards[6]
            # If returning reinforce getting closer to nest
            if nest_dist2(last_loc) > nest_dist2((nr,nc)):
                total_reward += rewards[7]
            # else:
                # total_reward += -rewards[7]
        return total_reward

    def step(self, action):
        antID = self.antIndex
        reward = self.get_reward(action)

        self.action_mem[antID] = action + (8 if self.has_food[antID] else 0)

        # Calculating Next States
        nr,nc = self.calc_next_location(action)
        self.ant_locations[antID] = (nr,nc)
        self.state.explored_space[nr,nc] = 1

        # Adding food and setting Trail
        if self.has_food[antID]:
            if (nr,nc) == self.nest:
                self.storeFood()
                self.has_food[antID] = 0
            else:
                self.state.trail_space[nr,nc] += 1
        # Setting food space if ant with no food walks over it
        elif self.state.food_space[nr,nc]:
            self.state.food_space[nr,nc] -= 1
            self.state.food_space[self.state.food_space<0] = 0
            self.has_food[antID] = 1

        # Needed to be calculated before state_mem modified potentially
        next_state = self.get_state()

        # If last ant, Update: State Memory, Trail Space, Done
        if self.antIndex >= len(self.ants) - 1:
            self.full_step_update()

        # Next Ant
        self.antIndex = (self.antIndex + 1) % len(self.ants)
        return next_state, reward, self.done

    def full_step_update(self):
        self.state_mem = np.concatenate((self.state_mem[:,1:],np.array(self.ant_locations).reshape(self.num_ants,1,2)),1)
        self.state.trail_space -= 0.05
        self.state.trail_space[self.state.trail_space<0] = 0
        self.done = self.totalFoodCollected == self.total_starting_food

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
            if not self.has_food[a.antID]:
                antFY.append(ar)
                antFX.append(ac)
            else:
                antNFY.append(ar)
                antNFX.append(ac)

        # Plot the nest, ants, trails
        plt.clf()
        if stepNum >= 0:
            plt.annotate('Step: '+str(stepNum),(5,-1))
        plt.annotate('Food Stored: '+str(self.totalFoodCollected),(9,-1))
        plt.annotate('Food Left: '+str(self.state.remaining_food()),(14,-1))
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

class AntAgent:
    def __init__(self,ID,env,ant_type='normal'):
        self.antID = ID
        self.env = env
        self.type = ant_type
        self.actions = ACTIONS

    def __call__(self,X):
        return self.policy(X)

    def policy(self, state):
        return [1/len(self.actions)]*len(self.actions)


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
        obstacles = F.pad(self.state.grid_space, (1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]
        food      = F.pad(self.state.food_space, (1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]
        trail     = F.pad(self.state.trail_space,(1,1,1,1),value=-1)[r:r+3,c:c+3].flatten()[actView]

        to_nest = torch.zeros(2)
        if   r < self.nest[0]: to_nest[0] =  1
        elif r > self.nest[0]: to_nest[0] = -1
        if   c < self.nest[1]: to_nest[1] =  1
        elif c > self.nest[1]: to_nest[1] = -1

        return np.concatenate((  food,
                                 trail,
                                 obstacles,
                                 torch.tensor([self.has_food[antID], self.action_mem[antID]]),
                                 torch.tensor(self.state_mem[antID]).flatten(), # state_mem tensor?
                                 torch.tensor(self.ant_locations[antID]),
                                 to_nest,
                               ))

    def calc_next_location(self,action,loc=None,avoid_obs=True):
        r,c = self.ant_locations[self.antIndex] if loc is None else loc
        dr,dc = self.actions[action]
        nr = min(max(r+dr,0),self.state.grid_space.shape[0]-1)
        nc = min(max(c+dc,0),self.state.grid_space.shape[0]-1)
        if avoid_obs and self.state.grid_space[nr,nc]:
            nr,nc = r,c
        return nr,nc


