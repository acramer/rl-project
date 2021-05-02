from Configure import parse_configs, print_configs
from procedural_ant import ProceduralEnvironment 

from centralized_q import CentralEnvironment, DecentralizedEnvironment

import numpy as np

Environments = {'procedural': ProceduralEnvironment,
                'central-q':  CentralEnvironment,
                'decentral-q': DecentralizedEnvironment,    # Please add this to configs.architecture
                }

def main(configs):
    num_ants = 10
    environment = Environments['decentral-q'](num_ants=configs.num_ants,epochs=configs.epochs,max_steps=configs.max_steps,epsilon=configs.epsilon)
    done = False
    for i in range(configs.max_steps):
        for ant in environment.ants:
            choice_dist = ant.policy(environment.get_state())
            action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
            state, reward, done = environment.step(action)
        if done: break
        if configs.simulate: environment.plot_environment(i)
    print('Total Food:',environment.totalFoodCollected)
    print('Last Step:',i)


if __name__=="__main__":
    configs = parse_configs()
    if configs.help:
        print_configs()
    else:
        main(configs)
