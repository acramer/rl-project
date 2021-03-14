from random import randint
import numpy as np

class ant_env:
    def __init__(self, env_size):
        self.space = np.zeros((env_size,env_size,))
        
    def __call__(self,X):
        raise NotImplemented
        return self.act(X)

    def act(self,X):
        raise NotImplemented

class simple_env(ant_env):
    def __init__(self, env_size, nest_loc='center', num_actors=1):
        super().__init__(env_size)
        self.food_space = self.space.copy()
        self.trail_space = self.space.copy()
        self.explore_space = self.space.copy()
        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        self.init_food()
        self.actors = [self.nest]*num_actors
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

        r,c = self.actors[idx]
        dr, dc = actions[action]
        nr = max(min(r+dr,0),self.space.shape[0])
        nc = max(min(c+dc,0),self.space.shape[0])
        self.actors[idx] = (nr,nc)
        if self.food_space[self.actors[idx]]: 
            self.food_space[self.actors[idx]] -= 1

    def getSpace(self,idx):
        r, c = self.actors[idx]
        trail = list(np.pad(self.trail_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        r, c = self.actors[idx]
        food = list(np.pad(self.food_space,(1,1),constant_values=-1)[r:r+3,c:c+3].flatten())
        return food[:4]+food[5:],trail[:4]+trail[5:]

    def __str__(self):
        R,C = self.space.shape
        ret = [['{:^5}'.format(str(self.food_space[r,c]) if self.food_space[r,c] else '.') for c in range(C)] for r in range(R)]
        ret[self.nest[0]][self.nest[1]] = '{:^5}'.format('N')
        for a in self.actors:
            if a != self.nest:
                ret[a[0]][a[1]] = '{:^5}'.format('x')
        return ''.join(['-----']*len(ret[0]))+'\n|'+'|\n|'.join([''.join(r) for r in ret])+'|\n'+''.join(['-----']*len(ret[0]))
        
def nAnts(n=10,env_size=20,episode_size=100):
    colony = colony(n)
    env = simple_env(env_size,num_actors=n)
    
    for _ in range(episode_size):
    # while not env.done:
        for i, ant in enumerate(colony):
            env.step(i,ant(env.getSpace(i)))


class colony:
    def __init__(self,n=10):
        self.ants = [ant() for _ in range(n)]

    def __iter__(self):
        return self.ants

class ant:
    def __init__(self,memory_size=15,exploring=False):
        self.memory = []
        self.action_memory = 0
        self.memory_size = memory_size
        self.exploring = exploring
        self.foraging = False
        def normal_dist(x , mean , sd):
            return 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        self.nd = normal_dist(np.array([0,45,90,135,180,-135,-90,-45])*np.pi/180,0,0.5)

    def __call__(self,X):
        return self.act(X)

    def act(self,X):
        act = 0
        self.action_memory = act
        return act

    def walk(self,X):
        food, trail = X
        alist = [[0,1,2,3,4,5,6,7],
                 [1,2,3,4,5,6,7,0],
                 [2,3,4,5,6,7,0,1],
                 [3,4,5,6,7,0,1,2],
                 [4,5,6,7,0,1,2,3],
                 [5,6,7,0,1,2,3,4],
                 [6,7,0,1,2,3,4,5],
                 [7,0,1,2,3,4,5,6],
                ]
        return np.random_choice(alist[self.action_memory],p=self.nd)    



def main():
    print(len('{:^5}'.format('3')))
    s = simple_env(11)
    print(s)
    s.actors[0] = (0,0)
    print(s)
    print(np.pad(s.food_space,(1,1),constant_values=-1))
    

if __name__ == '__main__':
    main()
