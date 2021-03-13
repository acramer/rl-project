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
    def __init__(self, env_size, nest_loc='center'):
        super().__init__(env_size)
        self.food_space = self.space.copy()
        self.trail_space = self.space.copy()
        self.explore_space = self.space.copy()
        if nest_loc == 'center':
            self.nest = ((env_size-1)//2,(env_size-1)//2)
        self.init_food()

    def init_food(self):
        np.random.normal(1,0,self.space)
        

class colony:
    def __init__(self,n=10):
        self.ants = [ant() for _ in range(n)]

    def __iter__(self):
        return self.ants

class ant:
    def __init__(self):
        pass
        self.id = randint(0,100)

    def __call__(self,X):
        raise NotImplemented
        return self.act(X)

    def act(self,X):
        raise NotImplemented


def main():
    nest = (4,4)
    nest_mask_size = 2
    val_range = (1,10)
    dist = np.random.normal(0,10,(10,10))
    space = np.zeros((10,10))
    for i in range(1,10):
        space += (dist.astype(np.int32) == i+1).astype(np.float32) * (i+1)
    nest_mask = np.ones((10,10))
    #nest_mask = 
    #space *= 
    print(space)
    #print((np.random.normal(0,2,(10,10)) > 0).astype(np.float32)*np.ones((10,10)))


if __name__ == '__main__':
    main()
