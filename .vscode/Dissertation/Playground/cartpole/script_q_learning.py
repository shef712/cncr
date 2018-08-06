import gym
import numpy as np 
import matplotlib.pyplot as plt 

gym.logger.set_level(40)

env = gym.make('CartPole-v0')

# there is 10^4 states because we have split each state variable, which was previous continuous within some range, into 10 discrete "bins" within some range, where the range will be limited to the fail state of this environment
# this allows us to go from a continuous space with infinite states to a discrete space with only 10,000 possible states
MAXSTATES = 10**4
# discount variable
GAMMA = 0.9
# learning rate (as an action-value steps closer to the current expected cumulative reward)
ALPHA = 0.01

# utility function for max value and key associated with this
# we retrieve the maximum valued element from dictionary d
# this will correspond to the best action (action with the reward) for a state 
# i.e. when max_v = max_dict(Q[state])
def max_dict(d):
	max_v = float('-inf')
	for key, val in d.items():
		if val > max_v:
			max_v = val
			max_key = key
	return max_key, max_v

# using numpy, we create a 4x10 matrix of zeroes, this will correspond to the 4 different quantities placed in their respective "bin", with 10 bins for each quantity
def create_bins():
	# obs[0] -> cart position --- -4.8 - 4.8
	# obs[1] -> cart velocity --- -inf - inf
	# obs[2] -> pole angle    --- -41.8 - 41.8
	# obs[3] -> pole velocity --- -inf - inf
	
	bins = np.zeros((4,10))
	bins[0] = np.linspace(-4.8, 4.8, 10) # cart position
	bins[1] = np.linspace(-5, 5, 10) # cart velocity, theoretically has an infinite range, but imposing this limitation remains feasible and considers temrination range too 
	bins[2] = np.linspace(-.418, .418, 10) # pole angle
	bins[3] = np.linspace(-5, 5, 10) # pole velocity

	# can alter the number of bin number and range (for infinite limits) which means we alter the number of states), trying varied values may achieve better results 

	return bins

# for a given state, we determine what bin it falls into
# so if we pass this method a number (observation) and a set of bins, numpy's "digitize" method will calculate which bin the observation quantity falls into
# e.g. bin 1 -> [0, 2) and bin 2 -> [2, 4), then if x - 3, the return bin value would be 2
# for each quantity in the observation array (4 elements), we will get the corresponding bin for each variable
# thus we transform the continuous space for a variable,into a discrete one
def assign_bins(observation, bins):
	state = np.zeros(4)
	for i in range(4):
		state[i] = np.digitize(observation[i], bins[i])
	return state

# structured the states as string, to work as the keys for the dictionary
def get_state_as_string(state):
	string_state = ''.join(str(int(e)) for e in state)
	return string_state

# we number each state from 0 to 9999, to correspond to each of the 10,000 states
# to maintain string length for a state, we execute "zfill(4)" which adds on trailing 0's
# for the first few stages, we obtain state string values as: '0000', '0001', '0002' which correspond to the first three stages
# returned array contains all stage indices, in the form of a four-letter string
# we then use these strings as keys for our dictionary, i.e. the q-value matrix, "Q"
def get_all_states_as_string():
	states = []
	for i in range(MAXSTATES):
		states.append(str(i).zfill(4))
	return states

# creating the q-value matrix, Q
# we create a row for each state and a column for each action available (constant of 2 actions possible for each state, i.e. move left or move right)
# intialise the matrix with 0, which corresponds to the reward for each action at each state
def initialize_Q():
	Q = {}

	all_states = get_all_states_as_string()
	for state in all_states:
		Q[state] = {}
		for action in range(env.action_space.n):
			Q[state][action] = 0
	return Q

# plays one game, eps is the probability that a random action is taken through epsilon-greedy policy, this will decay as time increases
def play_one_game(bins, Q, eps=0.5):
	observation = env.reset()
	done = False
	cnt = 0 # number of moves in an episode, episode terminates after 200 moves/timesteps
	state = get_state_as_string(assign_bins(observation, bins))
	total_reward = 0

	while not done:
		cnt += 1	
    
		# np.random.randn() seems to yield a random action 50% of the time ?
        # Normal has a single most likely value, uniform has every allowable value equally likely. Uniform has a piecewise constant density.
        # So uniform distribution is more applicable in this case
		if np.random.uniform() < eps:
			act = env.action_space.sample() # epsilon greedy
		else:			
			act = max_dict(Q[state])[0] # otherwise pick the maximum action
			# should there be another random factor when the top actions equal the same, because currently the first action from this set will always be picked, though i guess it wont matter because as soon as it is picked, it will either get higher or lower, and determine the state space explored... maybe these scenarios should be flagged, and both actions are always picked somehow... then again, this issue is only due to initialisation, and if an action is truly good then it will get a higher reward,this doesnt seem like a big issue after all
		
		observation, reward, done, _ = env.step(act)

		# reward is +1 for every timestep the pole is kept upright
		total_reward += reward

		if done and cnt < 200:
			# massive penalty if this action in this state causes the pole to fall over
			reward = -300

		# get the new state index
		state_new = get_state_as_string(assign_bins(observation, bins))

		# action is the key
		# Q[state_new] returns the set of actions available with each key corresponding to each action, the value corresponding to a key, in this dictionary, is the expected reward from this action
		# we want the action that returns the highest expected reward
		a1, max_q_s1a1 = max_dict(Q[state_new])

		# note we use this same value of maximum expected value, for updating the new value of the current state too
		Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])

		# original line of code below, but we do not seem to be using the new "act" variable, corresponding to the current best action, since we recalculate the best action for the new state
		state, act = state_new, a1	
		# we can now assign the current state to be the new state				
		#state = state_new					

	# the above is for one run of the cartpole simulation, many runs are needed to update the Q-matrix to get a clear picture of the best actions,with bad actions yield negative values due to the -300 penalty
	return total_reward, cnt

def play_many_games(bins, N=10000):
	# initalise Q to be used by the environment
	Q = initialize_Q()

	length = []
	reward = []
	for n in range(N):
		#eps=0.5/(1+n*10e-3)
		eps = 1.0 / np.sqrt(n+1)

		# get the total reward and number of timesteps completed for each run of the simulation
		episode_reward, episode_length = play_one_game(bins, Q, eps)
		
		if n % 100 == 0:
			print(n, '%.4f' % eps, episode_reward)
		length.append(episode_length)
		reward.append(episode_reward)

	return length, reward

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		# each element in the "running_avg" array contains the mean, of the 
		# e.g. t = 750, max(0, 750-100) = 650
		# running_avg[750] = np.mean(totalrewards[650: 751])
		# this will smooth out the graph, by calculating the mean across the previous 100 runs, or 0 if t < 100
		# i.e. creates a plot of the average of the last 100 games, rather than the plots for each single run, which may be extremely fluctuating
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

if __name__ == '__main__':
	# run simulation
	bins = create_bins()
	episode_lengths, episode_rewards = play_many_games(bins)

	# plot graph
	plot_running_avg(episode_rewards)