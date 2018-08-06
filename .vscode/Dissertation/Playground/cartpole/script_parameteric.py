import gym
import numpy as np
# allows for recording
from gym import wrappers

gym.logger.set_level(40)

env = gym.make('CartPole-v0')

# keep track of longest episode 
best_length = 0
# list of all episodes
episode_lengths = []
# keep track of weights corresponding to longest episode
best_weights = np.zeros(4)

# try 100 different parameters, and average these over 100 games
for i in range(100):
    # initialise new set of random weights, creating a 4-element array with values ranging from -1 to 1
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    # this will help the agent make more intelligent decisions on moving left or right

    # calculate a dot product between the weights and the observation, and if the result is negative the agent moves left and positive means the agent moves right

    # keep track of each episode with this random weight, averaged over 100 games
    length = []

    for j in range(100):
        observation = env.reset()
        
        done = False
        cnt = 0

        while not done:
            # env.render()
            
            cnt += 1

            action = 1 if np.dot(observation, new_weights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done: 
                break

        length.append(cnt)
    
    average_length = float(sum(length) / len(length))

    if average_length > best_length:
        best_length = average_length
        best_weights = new_weights

    episode_lengths.append(average_length)
    if i % 10 == 0:
        print('best length is ', best_length)

done = False
cnt = 0

# recording, pass in the environment,directory to save to and force overwrite
env = wrappers.Monitor(env, 'saved_replays/cartpole', force=True)

observation = env.reset()
while not done:
    # dont need to render because we are recording which calls "render()"
    env.render()
    
    cnt += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done: 
        break

print('with best weights the game lasted ', cnt, ' moves')



