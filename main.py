from Configure import parse_configs, print_configs
from procedural_ant import ProceduralEnvironment 
from centralized_q import CentralEnvironment, JointEnvironment, DecentralizedEnvironment, DeepCentralEnvironment

import numpy as np
import matplotlib.pyplot as plt

Environments = {'procedural': ProceduralEnvironment,
                'central-q':  CentralEnvironment,
                'joint-q': JointEnvironment,
                'dec-q': DecentralizedEnvironment,
                'deep-central-q': DeepCentralEnvironment,
                }

def main(configs):
    environment = Environments[configs.architecture](args=configs,num_ants=configs.num_ants,epochs=configs.epochs,max_steps=configs.max_steps,epsilon=configs.epsilon)
    done = False
    if isinstance(environment,DecentralizedEnvironment):
        configs.max_steps = 50000
    for i in range(configs.max_steps):
        if isinstance(environment,DecentralizedEnvironment):
            environment.train()
        else:
            for ant in environment.ants:
                choice_dist = ant.policy(environment.get_state())
                action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                state, reward, done = environment.step(action)
        if done: break
        if configs.simulate: environment.plot_environment(i)
        # environment.plot_environment(i)
    print('Total Food:',environment.totalFoodCollected)
    print('Left Food:',environment.state.remaining_food())
    print('Last Step:',i)
    if isinstance(environment,DecentralizedEnvironment):
        fig, ax = plt.subplots(2)
        ax[0].plot(environment.rewards)
        ax[1].plot(environment.left_food)
        plt.show()
        plt.pause(30)
    


if __name__=="__main__":
    configs = parse_configs()
    if configs.help:
        print_configs()
    else:
        main(configs)
