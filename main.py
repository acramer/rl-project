from Configure import parse_configs, print_configs
from envs.procedural_ant import ProceduralEnvironment 
from envs.centralized_q import CentralEnvironment, JointEnvironment, DecentralizedEnvironment
from envs.deep_central_q import DeepCentralEnvironment

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter
metadata = {'title':'ants simulation','artist':'ants','comment':'comments'}
writer = FFMpegWriter(fps=15,metadata=metadata)

Environments = {'procedural': ProceduralEnvironment,
                'central-q':  CentralEnvironment,
                'joint-q': JointEnvironment,
                'dec-q': DecentralizedEnvironment,
                'deep-central-q': DeepCentralEnvironment,
                }

# Given a directory of model folders of the form "<NUM>_<DESCRIPTION>", creates and returns a new folder
#   with the number incremented and description optionaly included.
def generate_model_id(directory, des=''):
    from os import walk, path, mkdir
    if not path.isdir(directory): mkdir(directory)

    def safe_int(i):
        try:
            return int(i)
        except (ValueError, TypeError):
            return -1

    model_nums = sorted(list(map(safe_int, list(map(lambda x: x.split('-')[0], list(walk(directory))[0][1])))))
    model_nums.insert(0, -1)

    description = str(model_nums[-1] + 1)
    if des: description += '-' + des

    mkdir(directory+'/'+description)
    return description

def plot_average(configs):
    configs.description = generate_model_id(configs.save_model_dir,configs.description)
    environment = Environments[configs.architecture](args=configs,num_ants=configs.num_ants,epochs=configs.epochs,max_steps=configs.max_steps,epsilon=configs.epsilon)
    done = False
    food_per_step   = [0]*configs.max_steps
    reward_per_step = [0]*configs.max_steps
    for ei in range(configs.epochs):
        for i in range(configs.max_steps):
            total_reward = 0
            for ant in environment.ants:
                choice_dist = ant.policy(environment.get_state())
                action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                state, reward, done = environment.step(action)
                total_reward += reward
            # reward_per_step[i] += (total_reward                   - reward_per_step[i])/(ei+1)
            # food_per_step[i]   += (environment.totalFoodCollected - food_per_step[i]  )/(ei+1)
            reward_per_step[i] += total_reward/environment.num_ants
            food_per_step[i] += environment.totalFoodCollected
            if done: break
    fig, ax = plt.subplots(2)
    ax[0].plot(reward_per_step)
    ax[1].plot(food_per_step)
    plt.show()
    plt.pause(30)

def train(configs):
    configs.description = generate_model_id(configs.save_model_dir,configs.description)
    environment = Environments[configs.architecture](args=configs,num_ants=configs.num_ants,epochs=configs.epochs,max_steps=configs.max_steps,epsilon=configs.epsilon)
    done = False
    if isinstance(environment,DecentralizedEnvironment):
        configs.max_steps = 50000
    if configs.save_video:
        fig = plt.figure()
        with writer.saving(fig,'videos/sim'+configs.architecture+configs.description+'.mp4',100):
            for i in range(configs.max_steps):
                for ant in environment.ants:
                    choice_dist = ant.policy(environment.get_state())
                    action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                    state, reward, done = environment.step(action)
                environment.plot_environment(i)
                writer.grab_frame()
                if done: break
    else:
        for i in range(configs.max_steps):
            if isinstance(environment,DecentralizedEnvironment):
                environment.train()
            else:
                for ant in environment.ants:
                    choice_dist = ant.policy(environment.get_state())
                    action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                    state, reward, done = environment.step(action)
            if configs.simulate: environment.plot_environment(i)
            if done: break
    print('Total Food:',environment.totalFoodCollected)
    print('Left Food:',environment.state.remaining_food())
    print('Last Step:',i)
    if isinstance(environment,DecentralizedEnvironment):
        fig, ax = plt.subplots(2)
        ax[0].plot(environment.rewards)
        ax[1].plot(environment.left_food)
        plt.show()
        plt.pause(30)
    
def main(configs):
    if configs.mode == 'average':
        plot_average(configs)
    else:
        train(configs)

if __name__=="__main__":
    configs = parse_configs()

    if configs.architecture=='dec-q':
        configs.max_steps = 50000

    if configs.help:
        print_configs()
    else:
        main(configs)
