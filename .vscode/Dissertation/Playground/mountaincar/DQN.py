import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

gym.logger.set_level(40)

class DQN:
    # constructor
    def __init__(self, env):
        # handle our environment
        self.env = env

        # deque - short for double-ended queue, allows for append/pop to either side of the data structure, only allows string values and works similar to lists
        # maxlen - Maximum size of a deque or None if unbounded.
        # memory variable serves as a key component of DQNs since trials are used continuously to train the model
        self.memory = deque(maxlen=2000)

        # discount factor
        self.gamma = 0.85

        # policy related
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # alpha
        self.learning_rate = 0.005
        
        # the rate of which we update our target network's weights
        self.tau = .125

        # intialise the two models 
        self.model        = self.create_model()
        self.target_model = self.create_model()

    # returns a model from our structured network
    def create_model(self):
        # create the model
        model   = Sequential()
        # get number of variables from state space, i.e. inputs to use as state representation
        state_shape  = self.env.observation_space.shape
        # defines the first layer, 12 inputs with the first layer having 24 neurons and "relu" activation function
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        # 48 neurons
        model.add(Dense(48, activation="relu"))
        # 24 neurons
        model.add(Dense(24, activation="relu"))
        # output neuron for each        
        model.add(Dense(self.env.action_space.n))

        # compile the model, using MSE cost function and gradient descent with learning rate
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        
        # return the created model
        return model

    # add to the memory 
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    # training our standard network
    def replay(self):
        batch_size = 32

        # if the memory is not big enough for a batch size yet, then do nothing and exit the method
        if len(self.memory) < batch_size: 
            return

         # get a random set of memory values of size "batch_size" to make up our sample
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample

            # use the target model to predict the "target q-values" for the state
            target = self.target_model.predict(state)

            # target will contain the q values of each action, where each action is enumerated from 0 to action_size_n, which we will have to manually construct

            # update the target q-values with the reward from the action experienced, so that the q-values used for target are as accurate as possible
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + self.gamma*Q_future
            
            # so we train our standard model with the up-to-date targets for how we want our agent to behave
            # the standard model is what our agent uses to predict q-values to decide what actions it will take
            # the target model is used to provide q-values for our network to use as targets and train with,
            # we do not want to obtain an updated q-value from the standard model (i.e. the immediate reward value from an action plus the long term reward of the next state) because we do not to want adjust weights for the same network whose goals (for optimum weights) are being changed so quickly from another sample input
            self.model.fit(state, target, epochs=1, verbose=0)
            # by setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch. 
            # verbose=0 will show you nothing (silent)
            # verbose=1 will show you an animated progress bar
            # verbose=2 will just mention the number of epoch


    # reorient goals where we simply copy over the weights from the standard model to the target one, but they are applied much more slowly, which is influencd by the "tau" variable
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            # target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            target_weights[i] = target_weights[i] + self.tau * (weights[i]-target_weights[i])
        self.target_model.set_weights(target_weights)

    # return action (enumerated) given state and accordance to the epsilon-greedy policy
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    # save the model to a file, which will allow "loadmodel()" to reinstantiate the model
    def save_model(self, fn):
        self.model.save(fn)

def main():
    # make our environment
    env     = gym.make("MountainCar-v0")

    # these two variables never get used by the looks of it
    # gamma   = 0.9
    # epsilon = .95

    trials  = 1000
    # amount of actions allowed in each trial (episode/trial length)
    trial_len = 500 

    # instantiate a DQN object
    dqn_agent = DQN(env=env)

    # create an empty list, possible indicators of progress to implement
    # steps = []

    for trial in range(trials):
        # reshape() is a numpy method that reshapes an array into the dimensions given
        # EXAMPLE a = np.zeros((10,2)); a.reshape(2,10), a.reshape(1,20)
        # get starting (current) state/observation from resetting the environment
        # the state is in the format (1x2) which i thought was default anyway...
        cur_state = env.reset().reshape(1,2)

        for step in range(trial_len):
            # for each timestep, we determine what action to take by taking the maximum q-value from the given state
            action = dqn_agent.act(cur_state)

            # get the new state, reward and done clause from stepping through this action
            new_state, reward, done, _ = env.step(action)

            # if we wanted to manually penalise the agent for every timestep it takes to achieve "done = True"
            # reward = reward if not done else -20
            
            new_state = new_state.reshape(1,2)
            
            # add the current variables and the new state and done clause to memory
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            # now that we've added a new memory to memory, we train our network with a batch of memories (which may or may not include the current timestep)
            dqn_agent.replay()       # internally iterates default (prediction) model

            # reorient our goal by copying across weight changes at a lower magnitude
            dqn_agent.target_train() # iterates target model

            # assign the new state as the current state
            cur_state = new_state

            # if done then break the timestep loop, no more actions are needed as we have completed/failed the environment
            if done:
                break
        
        # if the agent (network) fails to retrieve weights that solve the environment (mountain car reaching the flag) in a trial within 199 steps, then it counts as a fail
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            # save the network, usually when the trial length is at 500, so saves while it learns, so we do not need to start from scratch if training takes a while (should get some indicator of how well the agent is doing while running though)
            if step % 10 == 0:
                print("saving failed trial's model")
                dqn_agent.save_model("saved_models/trial-{}.model".format(trial))
        else:
            # agent has solved the environment within the allowed number of actions, we will save the model and break from doing any more trials... i guess we have a network that has weights that will solve this problem now... optimal though?
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("saved_models/success.model")
            break

if __name__ == "__main__":
    main()