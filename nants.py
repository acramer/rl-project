from random import randint
import numpy as np
from time import sleep

class ant_env:
    def __init__(self, env_size):
        self.space = np.zeros((env_size,env_size,))
        
    def __call__(self,X):
        raise NotImplemented
        return self.act(X)

    def act(self,X):
        raise NotImplemented

class SimpleEnv(ant_env):
    class SimpleActor:
        def __init__(self,loc):
            self.loc = loc
        def set(self,r,c):
            self.loc = (r,c)
        def get(self):
            return self.loc

    def __init__(self, env_size, nest_loc='center', num_actors=0,actors=None):
        super().__init__(env_size)
        self.food_space = self.space.copy()
        self.trail_space = self.space.copy()
        self.explore_space = self.space.copy()
        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        self.init_food()
        if actors:
            self.actors = actors
        else:
            self.actors = [SimpleEnv.SimpleActor(self.nest)]*num_actors

        self.full_step = 0
        self.done = False

    def init_food(self):
        #nest_mask_size
        nms = 2

        dist = np.random.normal(0,5,self.space.shape)
        for i in range(1,10):
            self.food_space += (dist.astype(np.int32) == i+1).astype(np.float32) * (i+1)

        # Only works for diagonal
        padding = (self.nest[0]-nms,self.space.shape[0]-(self.nest[0]+nms+1))
        nest_mask = np.pad(np.zeros((2*nms+1,2*nms+1)),padding,constant_values=(1))

        self.food_space *= nest_mask

    def step(self,idx,action):
        self.full_step += 1
        if self.full_step >= len(self.actors):
            self.trail_space -= (self.trail_space > 0).astype(np.float32)/5
            self.trail_space *= (self.trail_space > 1/5).astype(np.float32)
            self.full_step = 0

        actions = [ (-1, 0), # North
                    (-1, 1),
                    ( 0, 1), # West
                    ( 1, 1),
                    ( 1, 0), # South
                    ( 1,-1),
                    ( 0,-1), # East
                    (-1,-1), 
                    ]

        set_trail = bool(action//8)
        action %= 8

        r,c = self.actors[idx].get()
        dr, dc = actions[action]
        nr = min(max(r+dr,0),self.space.shape[0]-1)
        nc = min(max(c+dc,0),self.space.shape[0]-1)
        self.actors[idx].set(nr,nc)
        if self.food_space[self.actors[idx].get()]: 
            self.food_space[self.actors[idx].get()] -= 1
        self.trail_space[self.actors[idx].get()] += 1

    def getSpace(self,idx):
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        r, c = self.actors[idx].get()
        food = list(np.pad(self.food_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return convert_input(food), convert_input(trail)

    def __str__(self):
        R,C = self.space.shape
        ret = [['{:^5}'.format(str(self.food_space[r,c]) if self.food_space[r,c] else ('('+str(int(self.trail_space[r,c]+1))+')' if self.trail_space[r,c] else '')) for c in range(C)] for r in range(R)]
        ret[self.nest[0]][self.nest[1]] = '{:^5}'.format('N')
        for a in map(lambda x:x.get(),self.actors):
            if a != self.nest:
                ret[a[0]][a[1]] = '{:^5}'.format('x')
        return ''.join(['-----']*len(ret[0]))+'\n|'+'|\n|'.join([''.join(r) for r in ret])+'|\n'+''.join(['-----']*len(ret[0]))
        
def nAnts(n=10,env_size=20,episode_size=100):
    env = SimpleEnv(env_size)
    colony = Colony(env.nest,n)
    env.actors = colony.ants
    
    for _ in range(episode_size):
    # while not env.done:
        for i, ant in enumerate(colony):
            env.step(i,ant(env.getSpace(i)))
        sleep(0.25)
        print(env)
        print([('F' if a.foraging else 'R')+str(a.get()) for a in env.actors])
        print(colony.food)


class Colony:
    def __init__(self,nest,n=10):
        self.ants = [Ant(nest,self) for _ in range(n)]
        self.food = 0

    def __call__(self):
        self.food += 1

    def __iter__(self):
        return iter(self.ants)

class Ant:
    def __init__(self,nest,store_food,memory_size=5,exploring=False,mean=0,sd=0.5):
        self.nest = nest
        self.store_food = store_food
        self.location = nest
        self.exploring = exploring
        self.memory_size = memory_size

        # Random Walk Distribution
        rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
        normal_dist = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2)
        self.nd = list(normal_dist/np.sum(normal_dist))

        self.memory = []
        self.action_memory = 0
        self.foraging = False

    def __call__(self,X):
        return self.act(X)

    def set(self,r,c):
        self.location = (r,c)

    def get(self):
        return self.location

    def act(self,X):
        if len(self.memory) == self.memory_size+1:
            self.memory.pop(0)
        self.memory.append(self.location)

        actions = { (-1, 0):0, # North
                    (-1, 1):1,
                    ( 0, 1):2, # West
                    ( 1, 1):3,
                    ( 1, 0):4, # South
                    ( 1,-1):5,
                    ( 0,-1):6, # East
                    (-1,-1):7, 
                    }

        # actions in tuple form that take you in the direction of the nest
        to_nest = (0,0)
        if   self.location[0] < self.nest[0]: to_nest = ( 1, to_nest[1])
        elif self.location[0] > self.nest[0]: to_nest = (-1, to_nest[1])
        if   self.location[1] < self.nest[1]: to_nest = (to_nest[0],  1)
        elif self.location[1] > self.nest[1]: to_nest = (to_nest[0], -1)
        to_nest = [to_nest,(0,to_nest[1]),(to_nest[0],0)]

        food, trail = X

        if self.foraging:

            # Removing trail from spaces recently visited
            for a,v in actions.items():
                if (self.location[0]+a[0],self.location[1]+a[1]) in self.memory:
                    trail[v]=0
                if a in to_nest:
                    trail[v]=0

            if sum(food): 
                act = np.argmax(food)
                self.foraging = False
            elif sum(trail):
                act = np.argmax(trail)
            else:
                act = self.walk()
        else:
            if self.location == self.nest:
                self.store_food()
                self.foraging = True
                #self.action_memory = ?
                act = self.walk()
            else:
                for a,v in actions.items():
                    if a not in to_nest:
                        trail[v] = 0
                # adding preference for trails closer to home
                trail[list(actions.keys()).index(to_nest[0])] += 0.5
                act = np.argmax(trail) + 8
                
                
        self.action_memory = act // 8
        return act

    def walk(self):
        pdist = self.nd[self.action_memory:] + self.nd[:self.action_memory]
        return np.random.choice(8,p=pdist)



def main():
    # mean=0
    # sd=0.5
    # rads = np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180
    # nd =  list(1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((rads-mean)/sd)**2))
    # print(['{:9.4}'.format(n) for n in nd])
    # print(['{:9.4}'.format(n) for n in nd[1:]+nd[:1]])
    # print(['{:9.4}'.format(n) for n in nd[2:]+nd[:2]])
    # print(['{:9.4}'.format(n) for n in nd[3:]+nd[:3]])
    # print(['{:9.4}'.format(n) for n in nd[4:]+nd[:4]])
    # print(['{:9.4}'.format(n) for n in nd[5:]+nd[:5]])
    # print(['{:9.4}'.format(n) for n in nd[6:]+nd[:6]])
    # print(['{:9.4}'.format(n) for n in nd[7:]+nd[:7]])
    # print(['{:9.4}'.format(n) for n in nd[8:]+nd[:8]])
    # print()
    # print()
    # a = Ant()
    # for i in range(9):
    #     a.action_memory = i
    #     #print(['{:9.4}'.format(n) for n in a.walk()])
    #     print(a.walk())
    # print()
    # print()
    nAnts(10)

    #s = simple_env(11)
    #print(s)
    #s.actors[0].set(0,0)
    #print(s)
    #

if __name__ == '__main__':
    main()
