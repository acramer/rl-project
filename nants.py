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
        self.done = False

    def init_food(self):
        #nest_mask_size
        nms = 2

        dist = np.random.normal(0,10,self.space.shape)
        for i in range(1,10):
            self.food_space += (dist.astype(np.int32) == i+1).astype(np.float32) * (i+1)

        # Only works for diagonal
        padding = (self.nest[0]-nms,self.space.shape[0]-(self.nest[0]+nms+1))
        nest_mask = np.pad(np.zeros((2*nms+1,2*nms+1)),padding,constant_values=(1))

        self.food_space *= nest_mask

    def step(self,idx,action):
        actions = [ (-1, 0), # North
                    (-1, 1),
                    ( 0, 1), # West
                    ( 1, 1),
                    ( 1, 0), # South
                    ( 1,-1),
                    ( 0,-1), # East
                    (-1,-1), 
                    ]

        r,c = self.actors[idx].get()
        dr, dc = actions[action]
        nr = min(max(r+dr,0),self.space.shape[0]-1)
        nc = min(max(c+dc,0),self.space.shape[0]-1)
        self.actors[idx].set(nr,nc)
        if self.food_space[self.actors[idx].get()]: 
            self.food_space[self.actors[idx].get()] -= 1

    def getSpace(self,idx):
        r, c = self.actors[idx].get()
        food = list(np.pad(self.food_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return food[:4]+food[5:],trail[:4]+trail[5:]

    def __str__(self):
        R,C = self.space.shape
        ret = [['{:^5}'.format(str(self.food_space[r,c]) if self.food_space[r,c] else '.') for c in range(C)] for r in range(R)]
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


class Colony:
    def __init__(self,nest,n=10):
        self.ants = [Ant(nest) for _ in range(n)]

    def __iter__(self):
        return iter(self.ants)

class Ant:
    def __init__(self,nest,memory_size=15,exploring=False,mean=0,sd=0.5):
        self.nest = nest
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

        food, trail = X

        if self.foraging:

            # Removing trail from spaces recently visited
            for a,v in actions.items():
                if (self.location[1]+a[0],self.location[1]+a[1]) in self.memory:
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
                self.foraging = True
                #self.action_memory = ?
                act = self.walk()
            else:
                dr,dc = 0,0
                if   self.location[0] < self.nest[0]:
                    dr = 1
                elif self.location[0] > self.nest[0]:
                    dr = -1
                if   self.location[1] < self.nest[1]:
                    dc = 1
                elif self.location[1] > self.nest[1]:
                    dc = -1
                for a,v in actions.items():
                    if a not in [(dr,dc),(0,dc),(dr,0)]:
                        trail[v] = 0
                trail[list(actions.keys()).index((dr,dc))] += 0.5
                act = np.argmax(trail)
                
                
        self.action_memory = act
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
    nAnts()

    #s = simple_env(11)
    #print(s)
    #s.actors[0].set(0,0)
    #print(s)
    #

if __name__ == '__main__':
    main()
