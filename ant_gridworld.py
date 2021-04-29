from random import randint
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from Configure import parse_configs, print_configs

plt.ion()

class Env:
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

    def __init__(self,env_size=20, food_num=15, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10):
        self.reset(env_size, food_num, max_wt, nest_loc, nest_range, obstacle_no)  

    def reset(self,env_size=20, food_num=20, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10):
        self.state = Env.State(env_size)
        self.done = False
        self.ants = []
        self.expl_ants = []
        self.antIndex = 0
        self.totalFoodCollected = 0
        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))
        self.init_food_obstacles(food_num, max_wt, nest_range,obstacle_no)
        self.createColony(self.nest)

    def current_ant(self):
        if len(self.ants) > self.antIndex:
            return self.ants[self.antIndex]
        return None

    def next_ant(self):
        self.antIndex += 1
        self.antIndex %= len(self.ants)

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

    def createColony(self, nest, num_ants = 10):
        for idx in range(num_ants):
            if np.random.rand() > 0.15:
                self.addAntToColony(idx, nest, is_exploring=False)
            else:
                self.addAntToColony(idx, nest, is_exploring=True)

    def addAntToColony(self,idx, nest, is_exploring = "No_Arg" ):
        if is_exploring=="No_Arg":
            if len(self.expl_ants)/self.ants <= 0.15:
                is_exploring = True
            else:
                is_exploring = False

        ant = AntAgent(idx, self, nest, is_exploring)
        if is_exploring:
            self.expl_ants.append(ant)
        self.ants.append(ant)

    # def get_state(self, antID):
    def get_state(self):
        antID = self.current_ant().antID
        r, c  = self.ants[antID].get()
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        food  = list(np.pad(self.state.food_space, (1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return convert_input(food), convert_input(trail)

    # def get_reward(self, state, action):
    def get_reward(self, action):
        """ TODO
        as discussed, assign various rewards for various state,action pairs.
        """
        # state = self.get_state()
        # ant = self.current_ant()
        return 0

    def step(self, action):
        """ TODO depending on action, 
        1. Change food_space. trail_space
        2. Update the location of the ant
        """
        reward = self.get_reward(action)
        if self.antIndex >= len(self.ants) - 1:
            # trail_len = 15
            # self.trail_space -= (self.trail_space > 0).astype(np.float32)/trail_len
            # self.trail_space *= (self.trail_space > 1/trail_len).astype(np.float32)
            self.state.trail_space -= 0.05         # Pheromone evaporation rate: 0.05 per time step
            self.state.trail_space[self.state.trail_space<0] = 0
            self.done = self.state.remaining_food() == 0

        actions = [ (-1, 0), # North
                    (-1, 1),
                    ( 0, 1), # East
                    ( 1, 1),
                    ( 1, 0), # South
                    ( 1,-1),
                    ( 0,-1), # West
                    (-1,-1), ]

        set_trail = bool(action//8)
        pickup_food = bool(action//16)
        action %= 8

        current_ant = self.current_ant()
        r,c = current_ant.get()
        dr, dc = actions[action]
        nr = min(max(r+dr,0),self.state.grid_space.shape[0]-1)
        nc = min(max(c+dc,0),self.state.grid_space.shape[0]-1)
        # current_ant.set((nr,nc))
        current_ant.location = (nr,nc)
        if pickup_food and self.state.food_space[nr,nc]:
            self.state.food_space[nr,nc] -= 1
        # if set_trail and self.trail_space[self.actors[idx].get()] < 2:    # No upper limit on trail strength
        if set_trail and (nr,nc) != self.nest:      # No trail at nest
            self.state.trail_space[nr,nc] += 1            # Pheromone reinforcement

        state = self.get_state()
        self.next_ant()
        return state, reward, self.done


    def plot_environment(self):
        C,R = self.state.grid_space.shape
        oY, oX = np.where(self.state.grid_space>0)
        fY, fX = np.where(self.state.food_space>0)
        tY, tX = np.where(self.state.trail_space>0)
        # Foraging and non foraging ants 
        # antFY, antFX = [],[]
        # antNFY, antNFX = [],[]
        # for a in self.actors:
        #     ar,ac = a.get()
        #     if a.foraging: 
        #         antFY.append(ar)
        #         antFX.append(ac)
        #     else:
        #         antNFY.append(ar)
        #         antNFX.append(ac)

        # Plot the nest, ants, trails
        plt.clf()
        plt.axis([-1,C,R,-1])
        plt.scatter(self.nest[1],self.nest[0], color='#D95319', marker='s', s=70)   # Nest
        plt.scatter(tX,tY, color='#0072BD', s=4)                                    # Trail
        plt.scatter(fX, fY, color='#7E2F8E', s=15)
        plt.scatter(oX, oY, color='#000', s=15)                          # Food particles
        plt.scatter(*tuple(zip(*[a.get() for a in self.ants])), color='#77AC30', s=30)
        # plt.scatter(antFX, antFY, color='#77AC30', s=30)                            # Ants searching for food
        # plt.scatter(antNFX, antNFY, color='#A2142F', s=30)                          # Ants returning with food
        plt.xticks(range(-1,C,C//10))
        plt.yticks(range(-1,R,R//10))
        plt.show()
        plt.pause(0.01)

class AntAgent:
    def __init__(self,ID,env,nest,exploring=False, mean=0, sd=1):
        self.antID = ID
        self.nest = nest
        self.location = nest
        self.exploring = exploring
        self.env = env
        self.action_memory = 0
        self.foraging = False
        rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
        self.normal_dist = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2)

        self.actions = [ (-1, 0), # North
                         (-1, 1),
                         ( 0, 1), # East
                         ( 1, 1),
                         ( 1, 0), # South
                         ( 1,-1),
                         ( 0,-1), # West
                         (-1,-1), ]

    def get(self):
        return self.location

    # TODO: policy will be an object that can be called or function passed into ant, not a method
    def policy(self, state):
        # if len(self.memory) == self.memory_size+1:              # TODO: (self.memory) will be a state val
        #     self.memory.pop(0)
        # self.memory.append(self.location)

        # food, trail = X
        # if self.location == self.nest and not self.foraging:    # TODO: (self.location == self.nest) and (self.foraging -> has_food) will be a state vals
        #     self.store_food()
        #     self.foraging = True
        #     self.action_memory = randint(0,7)

        # if self.foraging:
        #     if self.exploring:
        #         act = self.explore(food,trail)
        #     else:
        #         act = self.forage(food,trail)
        # # Removing trail from spaces recently visited
        # else:
        #     to_nest = self.to_nest()                            # TODO: (to_nest) will be a fuzzy state val
        #     for v,a in enumerate(self.actions):
        #         # Remove trails that do not take the ant closer to the nest
        #         if a not in to_nest:
        #             trail[v] = 0
        #     # adding preference for trails closer to home
        #     trail[list(self.actions).index(to_nest[0])] += 0.5
        #     act = np.argmax(trail) + 8
        #         
        #         
        # self.action_memory = act % 8
        # return act
        return [1/len(self.actions)]*len(self.actions)
        # return [1]+[0]*(len(self.actions)-1)

    def get_action_probabilities(self):
        food, trail = self.env.get_state(self.antID)
        # TODO do neural network step here and get output
        """
        input -> food vector, trail vector
        output -> action vector 
        use normal distribution over output and find argmax
        depending on action vector, change location of the agent.
        """
        action_idx = 5
        action = self.actions[action_idx]
        return action

    def set(self, action):
        self.location = (self.location[0] + action[0],self.location[1] + action[1])


def run(configs):
    environment = Env()
    state = environment.reset()
    for _ in range(configs.max_steps):
        for ant in environment.ants:
            choice_dist = ant.policy(state)
            # print(ant.location,end=' ')
            action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
            # actions.append(action)
            new_state, reward, done = environment.step(action)
            # print(action,end=' ')
            # print(ant.location)
            state=new_state
        environment.plot_environment()


if __name__=="__main__":
    configs = parse_configs()
    run(configs)

    # print(environment.state.grid_space)
    # environment.plot_environment()
