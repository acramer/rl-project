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
    avg_food_collected_over_time   = np.zeros(configs.max_steps)
    avg_reward_collected_over_time = np.zeros(configs.max_steps)
    for ei in range(configs.epochs):
        food_coll = np.zeros(configs.max_steps)
        reward_coll = np.zeros(configs.max_steps)
        for i in range(configs.max_steps):
            total_reward = 0
            for ant in environment.ants:
                choice_dist = ant.policy(environment.get_state())
                action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                state, reward, done = environment.step(action)
                total_reward += reward
            food_coll[i:] = environment.totalFoodCollected
            reward_coll[i:] = total_reward/environment.num_ants

            if done or i+1==configs.max_steps:    
                environment.soft_reset()
                break
        avg_food_collected_over_time += food_coll
        avg_reward_collected_over_time += reward_coll

    fig, ax = plt.subplots(2)
    ax[0].plot(avg_food_collected_over_time/configs.epochs)
    ax[0].set(xlabel="Time step", ylabel="Average food collected")
    ax[1].plot(avg_reward_collected_over_time/configs.epochs)
    ax[1].set(xlabel="Time step", ylabel="Average reward obtained")
    fig.tight_layout()
    name = 'A-'+str(configs.architecture)+'-M-'+str(configs.max_steps)+'-E-'+str(configs.epochs)+'.png'
    fig.savefig(name)
    plt.show()
    plt.pause(3)

def train(configs):
    configs.description = generate_model_id(configs.save_model_dir,configs.description)
    environment = Environments[configs.architecture](args=configs,num_ants=configs.num_ants,epochs=configs.epochs,max_steps=configs.max_steps,epsilon=configs.epsilon)
    done = False
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
            for ant in environment.ants:
                choice_dist = ant.policy(environment.get_state())
                action = np.random.choice(list(range(len(choice_dist))),p=choice_dist)
                state, reward, done = environment.step(action)
            if configs.simulate: environment.plot_environment(i)
            if done: break
    print('Total Food:',environment.totalFoodCollected)
    print('Left Food:',environment.state.remaining_food())
    print('Last Step:',i)
    
def main(configs):
    if configs.mode == 'average':
        plot_average(configs)
    else:
        train(configs)

if __name__=="__main__":
    configs = parse_configs()

    if configs.help:
        print_configs()
    else:
        main(configs)
