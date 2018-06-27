
# Python module to find the LunarLander controller's optimal policy by training a CNN to retrieve the ship's coordinates.
#
# Gustavo Ciotto
# Lucas Rath Maia

from CMAES import CMAES
from collections import deque
from functools import reduce
from matplotlib import pyplot as plt
from matplotlib import patches
from PSO import Particle_Swarm_Opt
from sklearn.utils.extmath import softmax

import cv2
import deap
import gym
import multiprocessing
import numpy as np
import pickle

# Ignore annoying warnings
import warnings
warnings.filterwarnings("ignore")
gym.logger.set_level(40)



'''
 Lunar Lander documentation page
 Landing pad is always at coordinates (0,0). Coordinates are the first two numbers 
 in state vector. Reward for moving from the top of the screen to landing pad and 
 zero speed is about 100..140 points. If lander moves away from landing pad it loses 
 reward back. Episode finishes if the lander crashes or comes to rest, receiving 
 additional -100 or +100 points. Each leg ground contact is +10. Firing main engine 
 is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. 
 Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
 Four discrete actions available: do nothing, fire left orientation engine, fire 
 main engine, fire right orientation engine.

state = [
        (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
        (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
        vel.x*(VIEWPORT_W/SCALE/2)/FPS,
        vel.y*(VIEWPORT_H/SCALE/2)/FPS,
        self.lander.angle,
        20.0*self.lander.angularVelocity/FPS,
        1.0 if self.legs[0].ground_contact else 0.0,
        1.0 if self.legs[1].ground_contact else 0.0
        ]
'''

#%% Reinforcement Learning 

# Computes expected accumulative reward
def discount_and_normalize_rewards(episode_rewards, gamma, norm=False):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    if norm:
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    return discounted_episode_rewards


# Nx0  = number of episodes with different initial states to simulate each individual 
# Kmax = max episode timestep
# Nmc  = number of episodes for the same individual state for a stochastic policy (for deterministic Nmc=1)
# gamma= discount factor
def eval_fitness_simulate(env, part, Nx0=5, Kmax=1000, Nmc=1, gamma=0.95, show=False):
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n 
    
    Nx0_rewards = []
    for i_Nx0 in range(Nx0):
        Nmc_rewards = []
        for i_Nmc in range(Nmc):
            # env = save_point_env
            observation = env.reset()
            t_rewards = []
            for t in range(Kmax):
                if show:
                    env.render()
                part_matrix_w = np.reshape(part[:-action_space],(action_space,obs_space))
                part_matrix_b = part[-action_space:]
                prob_weights = softmax([part_matrix_w.dot(observation)+part_matrix_b]).ravel()
                # decide action
                action = np.random.choice(range(len(prob_weights)), p=prob_weights)
                observation, reward, done, info = env.step(action)
                t_rewards.append(reward)
                if done:
                    Nmc_rewards.append( discount_and_normalize_rewards( t_rewards, gamma, norm=False )[0] )
                    break
        Nx0_rewards.append( np.mean(Nmc_rewards) )
    return (np.mean(Nx0_rewards),)


def eval_fitness_simulate_CNN_Vision(env, part, sess, network, frames=3, Nx0=5, Kmax=1000, Nmc=1, gamma=0.95, show=False):
    obs_space = frames * 3
    action_space = env.action_space.n
    # part = np.random.random(action_space*obs_space+action_space)
    
    if show:
        fig,ax = plt.subplots(1)
        rect = patches.Rectangle((30,30), 50, 50, 0 ,linewidth=1,edgecolor='y',facecolor='none')
        ax.add_patch(rect)
        arrow = patches.Arrow(0,0,0,0)
        patcharrow = ax.add_patch(arrow)
        images = np.random.uniform(0, 255, size=(400, 600, 3))
        im = ax.imshow(images)
    
    Nx0_rewards = []
    for i_Nx0 in range(Nx0):
        Nmc_rewards = []
        for i_Nmc in range(Nmc):

            t_rewards = []
            observation_true = deque([env.reset()], maxlen=3)
            observation_pred = deque(maxlen=3)
            observation_img  = deque(maxlen=3)
            
            for t in range(Kmax):
                observation_img.append( env.render(mode='rgb_array')/255 )
                
                net_input = np.expand_dims( cv2.resize(observation_img[-1], dsize=(200, 200)) ,axis=0)
                net_prediction = sess.run( [network.y_pred], feed_dict={network.input_image:net_input} )                                
                new_pred = net_prediction[0].ravel()

                if t==0:    old_pred = new_pred
                else:       old_pred = observation_pred[-1]
                observation_pred.append( 0.7*( old_pred - new_pred ) + new_pred )
                
                if t % 10 and show:
                    im.set_data(observation_img[0].squeeze())                    
                    xpos = (observation_pred[0][0] +1)/2*600 - 25
                    ypos = (-1*observation_pred[0][1] + 1)*3/4 * 400 - 50
                    angle = observation_pred[0][2]
                    
                    rect.xy = tuple([xpos,ypos])                    
                    patcharrow.remove()
                    arrow = patches.Arrow(xpos+25,ypos+25,-60*np.cos(angle-np.pi/2),60*np.sin(angle-np.pi/2), 20)
                    patcharrow = ax.add_patch(arrow)                    
                    fig.canvas.draw()
                
                if t >= frames:
                    observation_frames = reduce((lambda x, y: np.concatenate([x,y])),observation_pred)
                    part_matrix_w = np.reshape(part[:-action_space],(action_space,obs_space))
                    part_matrix_b = part[-action_space:]
                    prob_weights = softmax([part_matrix_w.dot(observation_frames)+part_matrix_b]).ravel()
                    action = np.random.choice(range(len(prob_weights)), p=prob_weights)     # decide action
                else:
                    action = 0 #do nothing
                    
                observation, reward, done, info = env.step(action)
                observation_true.append(observation)
                t_rewards.append(reward)
                if done:
                    Nmc_rewards.append( discount_and_normalize_rewards( t_rewards, gamma, norm=False )[0] )
                    break
        Nx0_rewards.append( np.mean(Nmc_rewards) )
    env.close()
    return (np.mean(Nx0_rewards),)


def parallel_eval_fitness_simulate_CNN_Vision(part):   
    # Reinforcement Learning parameters
    
    Nx0=1
    Kmax = 1500
    Nmc  = 1
    gamma=1
    show=False
    
    # CNN parameters
    exp_name = '_2'   
    model_path = 'saved_models'+exp_name+'/'

    # Must be imported inside if we want to execute in more than one core.    
    from train_CNN_Vision import CNN_Lander
    import tensorflow as tf
    
    tf.reset_default_graph()
    network = CNN_Lander()
    saver = tf.train.Saver(max_to_keep=1)
    chkpoint = tf.train.latest_checkpoint(model_path)

    with tf.Session() as sess:
        saver.restore(sess, chkpoint)
        print('Model restored')
        reward = eval_fitness_simulate_CNN_Vision(env, part, sess, network, frames=3, Nx0=Nx0, Kmax=Kmax, Nmc=Nmc, gamma=gamma, show=show)
        return reward


def plot_logbook(logbook):
    plt.plot( np.arange(len(logbook))+1, [l['max'] for l in logbook] ,label='max')
    plt.plot( np.arange(len(logbook))+1, [l['min'] for l in logbook] ,label='min')
    plt.plot( np.arange(len(logbook))+1, [l['avg'] for l in logbook] ,label='avg')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)


#%% Main section

if __name__ == "__main__":
    
    # Start gym environment
    env = gym.make('LunarLander-v2')

    load = False  
    save = True
    CNN = True
    parallel = True
    
    version = '18_06_13_PolicySearch_CMAES_CNN_Vision_parallel_AGORA_VAI'
    
    # Evolution Strategy algorithm parameters
    population= 10
    GEN=1
    loop_GEN = 20
    if CNN:
        input_frames = 3
        size = input_frames * 3 * env.action_space.n + env.action_space.n
    else:
        size = env.observation_space.shape[0] * env.action_space.n + env.action_space.n # 36
    disp=True
    
    # Reinforcement Learning parameters
    Nx0=3
    Kmax = 1000
    Nmc  = 3
    gamma=1
       
    # CNN parameters
    if CNN and not parallel:
        import tensorflow as tf
        from train_CNN_Vision import CNN_Lander
        exp_name = '_2'
        model_path = 'saved_models'+exp_name+'/'
        tf.reset_default_graph()
        network = CNN_Lander()
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=1)
        chkpoint = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, chkpoint)

    if load:
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)
        with open(version+'.pkl', 'rb') as f:
            halloffame, logbook = pickle.load(f)
    else:
        if not parallel:
            if not CNN: 
                eval_fit = lambda part: eval_fitness_simulate(env, part, Nx0=Nx0, Kmax=Kmax, Nmc=Nmc, gamma=gamma, show=False)
            else:
                eval_fit = lambda part: eval_fitness_simulate_CNN_Vision(env, part, sess=sess, network=network, frames=input_frames, Nx0=Nx0, Kmax=Kmax, Nmc=Nmc, gamma=gamma, show=False)
            cmaes = CMAES(eval_fit, lambda_=population, size=size)
        else:
            cmaes = CMAES(parallel_eval_fitness_simulate_CNN_Vision, lambda_=population, size=size) 
            pool = multiprocessing.Pool(4)
            cmaes.toolbox.register('map', pool.map)
            print('Multiprocessing started')
            
        halloffame = []
        for _ in range(loop_GEN):
            print('111')
            cmaes.optimize(GEN, disp)
            print('222')
            halloffame.append( {'nGen':GEN, 'gene':cmaes.hof.items[0], 'fitness_value':cmaes.hof.keys[0].values[0]} )
        logbook = cmaes.logbook
        if save:    
            with open(version+'.pkl', 'wb') as output:
                save_object = [halloffame, logbook]
                pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
    
    plot_logbook(logbook)

