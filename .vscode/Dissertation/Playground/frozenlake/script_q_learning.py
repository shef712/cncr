import gym
import numpy as np 
import matplotlib.pyplot as plt 

gym.logger.set_level(40)

def main():
    env = gym.make('FrozenLake-v0')

    # Initialize table with all zeros
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    # We will then have a 16x4 matrix, one for each state-action pair

    # Set learning parameters
    ALPHA = .8
    GAMMA = .95

    num_episodes = 10000
    
    # Create lists to contain total reward per episode
    rewards = []
    
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        
        total_reward = 0
        done = False
        moves = 0

        #The Q-Table learning algorithm
        while moves < 99:
            moves+=1
            # Choose an action by greedily (with noise) picking from Q table
            act = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

            #Get new state and reward from environment
            new_state, reward, done, _ = env.step(act)

            #Update Q-Table with new knowledge
            Q[state, act] = Q[state, act] + ALPHA*(reward + GAMMA*np.max(Q[new_state,:]) - Q[state, act])
            
            total_reward += reward
            state = new_state
            
            if done:
                break
                
        rewards.append(total_reward)
    
        print ("Running average: ", str(sum(rewards)/num_episodes))
    
    # print ("Final Q-Table Values")
    # print (Q)

if __name__ == '__main__':
    main()
