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
            self.foraging = False
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
        elif nest_loc == 'corner':
            choices = ((0,0),(0,env_size-1),(env_size-1,0),(env_size-1,env_size-1))
            self.nest = choices[np.random.choice(len(choices))]
        elif nest_loc == 'random':
            self.nest = (np.random.choice(np.arange(env_size)),np.random.choice(np.arange(env_size)))
        self.init_food()

        if actors:
            self.actors = actors
        else:
            self.actors = [SimpleEnv.SimpleActor(self.nest)]*num_actors

        self.full_step = 0
        self.done = False

    def init_food(self):
        food_num = 10
        max_wt = 10
        foods = np.random.choice(np.arange(1,max_wt),food_num,replace=True)
        nest_range = 1
        nest_area =  [(i,j) for j in range(self.nest[1]-nest_range, self.nest[1]+nest_range+1) \
            if  j >= 0 and j < self.space.shape[1] \
                for i in range(self.nest[0]-nest_range, self.nest[0]+nest_range+1) \
                    if  i >= 0 and i < self.space.shape[0]]
        cand_points =  [(i,j) for i in np.arange(self.space.shape[0]) for j in np.arange(self.space.shape[1]) \
             if (i,j) not in nest_area]
        idx = np.random.choice(len(cand_points),food_num,replace=False)
        cand_points = [cand_points[i] for i in idx]
        for i in range(1,food_num):
            self.food_space[cand_points[i]] += foods[i]

    def step(self,idx,action):
        self.full_step += 1
        if self.full_step >= len(self.actors):
            trail_len = 15
            self.trail_space -= (self.trail_space > 0).astype(np.float32)/trail_len
            self.trail_space *= (self.trail_space > 1/trail_len).astype(np.float32)
            self.full_step = 0
            if not np.sum(self.food_space) and all([a.foraging for a in self.actors]):
                self.done = True

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
        pickup_food = bool(action//16)
        action %= 8

        r,c = self.actors[idx].get()
        dr, dc = actions[action]
        nr = min(max(r+dr,0),self.space.shape[0]-1)
        nc = min(max(c+dc,0),self.space.shape[0]-1)
        self.actors[idx].set(nr,nc)
        if pickup_food and self.food_space[self.actors[idx].get()]:
            self.food_space[self.actors[idx].get()] -= 1
        if set_trail and self.trail_space[self.actors[idx].get()] < 2:
            self.trail_space[self.actors[idx].get()] += 1

    def getSpace(self,idx):
        convert_input = lambda x : x[1:3]+x[5:6]+x[8:5:-1]+x[3:4]+x[0:1]
        r, c = self.actors[idx].get()
        food = list(np.pad(self.food_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        trail = list(np.pad(self.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return convert_input(food), convert_input(trail)

    def __str__(self):
        R,C = self.space.shape
        #ret = [['{:^5}'.format(str(self.food_space[r,c]) if self.food_space[r,c] else ('('+str(int(self.trail_space[r,c]+1))+')' if self.trail_space[r,c] else '')) for c in range(C)] for r in range(R)]
        ret = [['{:^5}'.format('O' if self.food_space[r,c] else ('.' if self.trail_space[r,c] else '')) for c in range(C)] for r in range(R)]
        ret[self.nest[0]][self.nest[1]] = '{:^5}'.format('N')
        for a in self.actors:
            ar,ac = a.get()
            if (ar,ac) != self.nest:
                ret[ar][ac] = '{:^5}'.format('e' if a.exploring else 'x')
        return ''.join(['-----']*len(ret[0]))+'\n|'+'|\n\n|'.join([''.join(r) for r in ret])+'|\n'+''.join(['-----']*len(ret[0]))
        
def nAnts(n=10,ne=0,env_size=20,episode_size=100,nest_loc='center'):
    env = SimpleEnv(env_size,nest_loc=nest_loc)
    colony = Colony(env.nest,n,ne)
    env.actors = colony.ants
    
    #for _ in range(episode_size):
    while not env.done:
        for i, ant in enumerate(colony):
            env.step(i,ant(env.getSpace(i)))
        sleep(0.1)
        print(env)
        #print('Food Count:',colony.food,[(('E' if a.exploring else 'F') if a.foraging else 'R')+str(a.get()) for a in env.actors])
        print('Food Count:',colony.food)


class Colony:
    def __init__(self,nest,n=10,nexp=1):
        self.ants = [Ant(nest,self) for _ in range(n-nexp)]+[Ant(nest,self,exploring=True) for _ in range(nexp)]
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

        self.actions = { (-1, 0):0, # North
                         (-1, 1):1,
                         ( 0, 1):2, # West
                         ( 1, 1):3,
                         ( 1, 0):4, # South
                         ( 1,-1):5,
                         ( 0,-1):6, # East
                         (-1,-1):7, 
                        }



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

        food, trail = X
        if self.location == self.nest:
            self.store_food()
            self.foraging = True
            self.action_memory = randint(0,7)

        if self.foraging:
            if self.exploring:
                act = self.explore(food,trail)
            else:
                act = self.forage(food,trail)
            # Removing trail from spaces recently visited
        else:
            to_nest = self.to_nest()
            for a,v in self.actions.items():
                if a not in to_nest:
                    trail[v] = 0
            # adding preference for trails closer to home
            trail[list(self.actions.keys()).index(to_nest[0])] += 0.5
            act = np.argmax(trail) + 8
                
                
        self.action_memory = act % 8
        return act

    def explore(self,food,trail):
        if max(food) > 0:
            efood = []
            for f,t in zip(food,trail):
                if f < 0:
                    efood.append(0)
                elif f > 0 and t > 0:
                    efood.append(0.1)
                elif f > 0:
                    efood.append(f)
                else:
                    efood.append(0.5)
            efood = list(np.array(efood)/sum(efood))
            act = np.random.choice(8,p=efood)
            if food[act]:
                act += 16
                self.foraging = False
        elif max(trail) > 0: 
            for i,t in enumerate(trail):
                if t < 0:
                    trail[i] = 10
            for a,v in self.actions.items():
                if (self.location[0]+a[0],self.location[1]+a[1]) in self.memory:
                    trail[v] = 9
                if a in self.to_nest():
                    trail[v] = 9
            act = np.argmin(trail)
        else:
            act = self.walk()
            if food[act] < 0:
                act += 4
                act %= 8

        return act

    def forage(self,food,trail):
        for a,v in self.actions.items():
            if (self.location[0]+a[0],self.location[1]+a[1]) in self.memory:
                trail[v] = min(1/13,trail[v])
            if a in self.to_nest():
                trail[v] = 0

        if max(food) > 0: 
            act = np.argmax(food) + 16
            self.foraging = False
        elif max(trail) > 0: 
            act = np.argmax(trail)
        else:
            act = self.walk()
            if food[act] < 0:
                act += 4
                act %= 8

        return act

    def to_nest(self):
        # actions in tuple form that take you in the direction of the nest
        to_nest = (0,0)
        if   self.location[0] < self.nest[0]: to_nest = ( 1, to_nest[1])
        elif self.location[0] > self.nest[0]: to_nest = (-1, to_nest[1])
        if   self.location[1] < self.nest[1]: to_nest = (to_nest[0],  1)
        elif self.location[1] > self.nest[1]: to_nest = (to_nest[0], -1)
        return [to_nest,(0,to_nest[1]),(to_nest[0],0)]

    def walk(self):
        pdist = self.nd[-1*self.action_memory:] + self.nd[:-1*self.action_memory]
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
        # Random Walk Distribution
    nAnts(30,10,env_size=25,nest_loc='random')

    #nest = (0,0)
    #space = np.zeros((10,10))
    #food_space = space.copy()
    #food_num = 20
    #max_wt = 5
    #foods = np.random.choice(np.arange(1,max_wt),food_num,replace=True)
    #nest_range = 2
    #nest_area =  [(i,j) for j in range(nest[1]-nest_range, nest[1]+nest_range+1) \
    #    if  j >= 0 and j < space.shape[1] \
    #        for i in range(nest[0]-nest_range, nest[0]+nest_range+1) \
    #            if  i >= 0 and i < space.shape[0]]
    #cand_points =  [(i,j) for i in np.arange(space.shape[0]) for j in np.arange(space.shape[1]) \
    #     if (i,j) not in nest_area]
    #idx = np.random.choice(len(cand_points),food_num,replace=False)
    #cand_points = [cand_points[i] for i in idx]
    #for i in range(1,food_num):
    #    food_space[cand_points[i]] += foods[i]
    #print(food_space)

    #s = simple_env(11)
    #print(s)
    #s.actors[0].set(0,0)
    #print(s)
    #

if __name__ == '__main__':
    main()
