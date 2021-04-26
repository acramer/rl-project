from random import randint
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

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

    def __init__(self,env_size=20, food_num=20, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10):
        self.reset(env_size, food_num, max_wt, nest_loc, nest_range, obstacle_no)

    def reset(self,env_size=20, food_num=20, max_wt=5, nest_loc="center", nest_range=1, obstacle_no=10):
        self.state = Env.State(env_size)

        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))
        self.init_food_obstacles(food_num, max_wt, nest_range,obstacle_no)

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
        self.state.add_food(food_locs, foods)

    def step(self, action):
        pass

    def get_reward(self, state, action):
        pass

    def take_action(self):
        pass

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
        # plt.scatter(antFX, antFY, color='#77AC30', s=30)                            # Ants searching for food
        # plt.scatter(antNFX, antNFY, color='#A2142F', s=30)                          # Ants returning with food
        plt.xticks(range(-1,C,C//10))
        plt.yticks(range(-1,R,R//10))
        plt.show()
        plt.pause(0.01)




if __name__=="__main__":
    environment = Env()
    # print(environment.state.grid_space)
    environment.plot_environment()