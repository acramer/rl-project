from random import randint
import numpy as np
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from collections import defaultdict

plt.ion()

class Env:
    class State:
        def __init__(self, env_size):
            self.grid_space = np.zeros((env_size, env_size))
            # self.env_shape = self.gridspace.shape
            self.border = np.pad(self.grid_space,(1,1),constant_values=-1)
            self.food_space = np.zeros((env_size, env_size))
            self.trail_space = np.zeros((env_size, env_size))
            self.explored_space = np.zeros((env_size, env_size))
            self.obstacle_locs = set()
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
                    self.obstacle_locs.add(loc)

    def __init__(self,env_size=20, food_num=15, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10, num_ants=3):
        self.reset(env_size, food_num, max_wt, nest_loc, nest_range, obstacle_no, num_ants)  

    def reset(self,env_size=20, food_num=20, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10,num_ants=3):
        self.state = Env.State(env_size)
        self.epsilon = 0.5
        self.alpha = 0.1
        self.gamma = 0.9
        self.done = False
        self.ants = []
        self.expl_ants = []
        self.totalFoodCollected = 0
        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))
        self.init_food_obstacles(food_num, max_wt, nest_range,obstacle_no)
        self.createColony(self.nest, num_ants=num_ants)

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

        cand_points = [i for i in cand_points if i not in self.state.obstacle_locs]
        idx = np.random.choice(len(cand_points),food_num,replace=False)
        food_locs = [cand_points[i] for i in idx]
        self.food_locs = food_locs
        self.state.add_food(food_locs, foods)

    def createColony(self, nest, num_ants = 3):
        for idx in range(num_ants):
            is_exploring=True
            self.addAntToColony(idx, nest, is_exploring)
            # if np.random.rand() > 0.15:
            #     self.addAntToColony(idx, nest, is_exploring=False)
            # else:
            #     self.addAntToColony(idx, nest, is_exploring=True)

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

    def get_state(self, antID):
        r, c  = self.ants[antID].get()
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        food = list(np.pad(self.state.food_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.state.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        obstacles = list(np.pad(self.state.grid_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return convert_input(food), convert_input(trail), convert_input(obstacles)

    def step(self, action, ant):
        """ TODO depending on action,
        1. Change food_space. trail_space
        2. Update the location of the ant
        3. Call get_reward for the action taken

        X. Check if all food is exhausted -> set self.done=True
        """
        # ant = self.ants[antID]
        r, c = ant.get()
        ant_loc = (r+action[0], c+action[1])
        total_reward = 0
        if (ant_loc[0] >= self.state.env_size) or (ant_loc[0] < 0) or \
                (ant_loc[1] >= self.state.env_size) or (ant_loc[1] < 0):
            total_reward += -35
            return ant_loc, total_reward, self.done
        else:
            food_val, trail_val, obstacle, is_explored = self.state.food_space[ant_loc],self.state.trail_space[ant_loc],self.state.grid_space[ant_loc], self.state.explored_space[ant_loc]

        if obstacle>0 :
            total_reward += -5
        else:
            ant.set(action)
            if ant.is_exploring:
                if ant.is_foraging:
                    if is_explored==0:  
                        total_reward += 1
                        self.state.explored_space[ant_loc] = 1
                    if food_val>0:
                        total_reward += 10
                        ant.is_foraging = False
                        self.state.food_space[ant_loc] -= 1
                    elif trail_val>0:   total_reward += 1
                else:
                    if ant_loc == self.nest:
                        total_reward += 100
                        ant.is_foraging = True
                        self.storeFood()
                    else:
                        dist_old = np.linalg.norm(np.array(self.nest) - np.array((r,c)))
                        dist_new = np.linalg.norm(np.array(self.nest) - np.array(ant_loc))
                        if dist_new<dist_old: total_reward += +5
                        else: total_reward += -1
                        self.state.trail_space[ant_loc] += 1
            # else:
            #     if ant.is_foraging:
            #         if is_explored==0:  
            #             total_reward += 0
            #             self.state.explored_space[ant_loc] = 1
            #         if food_val>0:
            #             total_reward += 5
            #             ant.is_foraging = False
            #             self.state.food_space[ant_loc] -= 1
            #         elif trail_val>0:   total_reward += 1
            #     else:
            #         if ant_loc == self.nest:
            #             total_reward += 1000
            #             ant.is_foraging = True
            #             self.storeFood()
            #         elif trail_val>0:
            #             l2_dist = np.linalg.norm(np.array(self.nest) - np.array(ant_loc))
            #             total_reward += -1*(l2_dist/4)
            #             total_reward += -1
            #             self.state.trail_space[ant_loc] += 1
            #         else:
            #             l2_dist = np.linalg.norm(np.array(self.nest) - np.array(ant_loc))
            #             total_reward += -1*(l2_dist/4)
            #             total_reward += -2
        if np.sum(self.state.food_space) == 0:
            self.done = True
        return ant_loc, total_reward, self.done

    def plot_environment(self):
        C,R = self.state.food_space.shape
        borderY, borderX = np.where(self.state.border<0)
        oY, oX = np.where(self.state.grid_space>0)
        fY, fX = np.where(self.state.food_space>0)
        tY, tX = np.where(self.state.trail_space>0)
        # Foraging and non is_foraging ants 
        antFY, antFX = [],[]
        antNFY, antNFX = [],[]
        for a in self.ants:
            ar,ac = a.get()
            if a.is_foraging: 
                antFY.append(ar)
                antFX.append(ac)
            else:
                antNFY.append(ar)
                antNFX.append(ac)

        # Plot the nest, ants, trails
        plt.clf()
        # plt.figure(figsize=(12,12))
        plt.axis([-2,C+1,R+1,-2])
        plt.scatter(self.nest[1],self.nest[0], color='#D95319', marker='s', s=70)   # Nest
        plt.scatter(tX,tY, color='#0072BD', s=4)                                    # Trail
        plt.scatter(fX, fY, color='#7E2F8E', s=15)
        plt.scatter(oX, oY, color='#000', s=15)       
        plt.scatter(borderX-1, borderY-1, color='#000', marker='s', s=70)                         
        plt.scatter(antFX, antFY, color='#77AC30', s=30)                            # Ants searching for food
        plt.scatter(antNFX, antNFY, color='#A2142F', s=30)                          # Ants returning with food
        plt.xticks(range(-1,C,C//10))
        plt.yticks(range(-1,R,R//10))
        plt.show()
        plt.pause(0.001)

class AntAgent:
    def __init__(self,ID,env,nest,exploring=False, mean=0, sd=1):
        self.antID = ID
        self.nest = nest
        self.location = nest
        self.is_exploring = exploring
        self.env = env
        self.action_memory = 0
        self.is_foraging = True
        self.my_rewards = []
        rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
        self.normal_dist = list(1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2))

        self.actions = [ (-1, 0), # North
                         (-1, 1),
                         ( 0, 1), # East
                         ( 1, 1),
                         ( 1, 0), # South
                         ( 1,-1),
                         ( 0,-1), # West
                         (-1,-1), ]
        self.nA = len(self.actions)


        self.Qval_forage = defaultdict(lambda: np.zeros(self.nA))
        self.Qval_return = defaultdict(lambda: np.zeros(self.nA))

    def get(self):
        return self.location

    def set(self, action):
        self.location = (self.location[0] + action[0],self.location[1] + action[1])

    def argmax(self, q_values):
            """
            Takes in a list of q_values and returns the index of the item 
            with the highest value. Breaks ties randomly.
            returns: int - the index of the highest value in q_values
            """
            top_value = float("-inf")
            ties = []
            
            for i in range(len(q_values)):
                if q_values[i] > top_value:
                    top_value = q_values[i]
                    ties = [i]
                elif q_values[i]==top_value:
                    ties.append(i)
            return np.random.choice(ties)
    def get_action_probabilities(self):
        # food, trail, obstacles = self.env.get_state(self.antID)
        # TODO do neural network step here and get output
        """
        input -> food vector, trail vector
        output -> action vector 
        use normal distribution over output and find argmax
        depending on action vector, change location of the agent.
        """
        state = self.get()
        action_probs = np.ones(self.nA, dtype=float) * self.env.epsilon / self.nA
        if self.is_foraging:
            best_action = self.argmax(self.Qval_forage[state])
        else:
            best_action = self.argmax(self.Qval_return[state])
        action_probs[best_action] = 1 - self.env.epsilon + self.env.epsilon / self.nA


        # print("BORDER", self.location, action_probs)
        # action_probs = action_probs/sum(action_probs)
        
        pdist = self.normal_dist[-1*self.action_memory:] + self.normal_dist[:-1*self.action_memory]
        action_probs = action_probs * pdist
        action_probs = action_probs/sum(action_probs)
        self.action_memory = np.random.choice(np.arange(self.nA),p=action_probs)
        return self.action_memory

        # action_idx = 5
        # action = self.actions[action_idx]
        # return action


def run():
    env = Env(num_ants=5)
    n = 0
    total_reward = 0
    while not env.done:
        for ant in env.ants:
            state = ant.get()
            action = ant.get_action_probabilities()
            next_state, reward, _ = env.step(ant.actions[action], ant)
            ant.my_rewards.append(reward)
            if ant.is_foraging:
                ant.Qval_forage[state][action] += env.alpha*(reward + \
                    env.gamma*np.max(ant.Qval_forage[next_state]) - ant.Qval_forage[state][action])
            else:
                ant.Qval_return[state][action] += env.alpha*(reward + \
                    env.gamma*np.max(ant.Qval_return[next_state]) - ant.Qval_return[state][action])
            total_reward += reward
        print("Mean Rewardzz", total_reward/len(env.ants))
        env.plot_environment()
        env.state.trail_space = env.state.trail_space-0.07
        env.state.trail_space[env.state.trail_space<0] = 0
        # n+=1
        # if n>50:
        #     break


if __name__=="__main__":
    run()

    # print(environment.state.grid_space)
    # environment.plot_environment()et