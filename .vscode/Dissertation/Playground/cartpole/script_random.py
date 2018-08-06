import gym
import numpy as np

# Hide the warning ("WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype.") by executing gym.logger.set_level(40). This will set minimal level of logger message to be printed to 40. Correspondingly, only error level messages will be displayed now.
gym.logger.set_level(40)

# instantiate a gym environment
env = gym.make('CartPole-v0')

# manage the epsiode length
done = False
cnt = 0

# have to reset the environment before using it
observation = env.reset()
# this will give us our first observation at timestep t = 0
# this is a 4-element array containing the markov state, which captures the environment, contains: pole position, pole velocity, cart position, cart velocity

while not done:
    # render the environment to be displayed, should be turned off to improve computation time
    env.render('mode=human')

    cnt += 1

    # we can sample a random action from the action space
    action = env.action_space.sample()

    # take the action for the next timestep
    observation, reward, done, _ = env.step(action)
    # returns:
    # observation - the new environment state after the action has been executed
    # reward - the reward of executing the action, which defined as "+1" for each timestep the terminating state has not been reached
    # done - flag variable if the episode has terminated
    # _ - placeholder for an info variable for debugging

    if done: 
        break

print('game lasted ', cnt, ' moves')



