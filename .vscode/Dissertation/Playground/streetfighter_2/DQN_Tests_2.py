import gym
import retro
import numpy as np
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import math 
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

gym.logger.set_level(40)

actions = [
    # directions only
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],

    # combat only
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    
    # direction + combat
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

actions_memory_n = 4
states_memory_n = 4
single_state_n = 14
state_n = ((states_memory_n + 1) * single_state_n) + actions_memory_n

class DQN:
    def __init__(self):
        self.memory = deque(maxlen=1000)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        
        self.learning_rate = 0.015
        self.tau = 0.2

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        first_layer_n = int((state_n + len(actions))/2)
        model.add(Dense(first_layer_n, input_dim=state_n, activation="relu"))
        model.add(Dense(len(actions)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        # batch_size = 32
        batch_size = 100

        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + self.gamma*Q_future
            
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i] + self.tau * (weights[i]-target_weights[i])
        self.target_model.set_weights(target_weights)

    def act(self, state):
        # self.epsilon *= self.epsilon_decay
        # self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randint(0,len(actions) - 1)
        return np.argmax(self.model.predict(state)[0])

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model = load_model(fn)

    def save_weights(self, fn):
        self.model.save_weights(fn)

    def load_weights(self, fn):
        self.model.load_weights(fn)

    def pre_process(self, info):
        # define variable limits
        min_x = 33554487 # LHS
        max_x = 33554889 # RHS
        min_y = 0 # peak of jump
        max_y = 192 # ground
        range_x = max_x - min_x
        range_y = max_y - min_y
        min_health = 0 # "<= 0" means death
        max_health = 176
        min_clock = 0
        max_clock = 99

        # get state variables
        agent_x = info["agent_x"]
        agent_y = info["agent_y"]
        enemy_x = info["enemy_x"]
        enemy_y = info["enemy_y"]
        agent_health = info["agent_health"]
        enemy_health = info["enemy_health"]
        clock = info["clock"]
        agent_combat_active = info["agent_combat_active"]
        enemy_combat_active = info["enemy_combat_active"]
        
        # calculate the remaining state variables
        agent_crouch = int(agent_x < min_x)
        agent_jump = int(agent_y < max_y)
        enemy_crouch = int(enemy_x < min_x)
        enemy_jump = int(enemy_y < max_y)
        
        # these variables indicate when we are already using normalised values
        normalised_agent_x = False
        normalised_enemy_x = False

        # initial values
        initial_agent_x = 0.373134328
        initial_enemy_x = 0.626865672

        # clean up
        if agent_x < min_x or agent_x > max_x:
            normalised_agent_x = True
            if (len(self.memory) == 0):
                agent_x = initial_agent_x
            else:
                agent_x = self.memory[-1][0][0][0]
        if enemy_x < min_x or enemy_x > max_x:
            normalised_enemy_x = True
            if (len(self.memory) == 0):
                enemy_x = initial_enemy_x
            else:
                enemy_x = self.memory[-1][0][0][2]

        if agent_health < min_health:
            agent_health = min_health
        if enemy_health < min_health:
            enemy_health = min_health
        if clock > max_clock:
            clock = max_clock

        if agent_combat_active == 512:
            agent_combat_active = 0
        elif agent_combat_active == 513:
            agent_combat_active = 1
        if enemy_combat_active == 512:
            enemy_combat_active = 0
        elif enemy_combat_active == 513:
            enemy_combat_active = 1

        # normalise
        agent_x = (agent_x - min_x) / range_x if not normalised_agent_x else agent_x
        agent_y = (agent_y - min_y) / range_y
        enemy_x = (enemy_x - min_x) / range_x if not normalised_enemy_x else enemy_x
        enemy_y = (enemy_y - min_y) / range_y
        agent_health = agent_health / max_health
        enemy_health = enemy_health / max_health
        clock = clock / max_clock
        absolute_diff = math.sqrt( ((agent_x - enemy_x)**2) + ((agent_y - enemy_y)**2) )

        state = [agent_x, agent_y, enemy_x, enemy_y, 
        agent_health, enemy_health, 
        agent_jump, enemy_jump, agent_crouch, enemy_crouch, 
        absolute_diff, clock, 
        agent_combat_active, enemy_combat_active]

        return state
    
def train_network():
    # train on the first three
    characters = ['Guile', 'Ken', 'ChunLi', 'Guile']
    average_won = []
    random_average_won = []

    dqn_agent = DQN()
    for c in range(len(characters)):
        state_name = 'Champion.MBisonVs' + characters[c]
        env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state=state_name)

        directional_frame_window = 20
        max_actions = 20000

        trials = 50
        random_trial = 0
        if c == 3:
            # going to run 30 test trials for trained agent and random agent, we expect poor performance for both
            trials = 60
            random_trial = 30

        agent_health = 176
        enemy_health = 176

        trial_n = []
        total_rewards = []
        rounds_won = []
        matches_won = []

        random_trial_n = []
        random_total_rewards = []
        random_rounds_won = []
        random_matches_won = []
        
        wins = 0

        if not c == 3:
            dqn_agent.epsilon = 1

        for trial in range(trials):
            cur_state = env.reset()
            states_memory = [ [0]*single_state_n ] * states_memory_n
            actions_memory = [0]*actions_memory_n
            intial_state = [0.373134328,1,0.626865672,1, 1,1, 0,0,0,0, 0.253731344,1, 0,0]
            for i in range(state_n - len(intial_state)):
                intial_state.append(0)
            cur_state = (np.array(intial_state)).reshape(1, state_n)

            action_step = 0
            total_reward = 0
            done = False

            new_clock = 0
            agent_stun = 0
            continuetimer = 0
            agent_rounds_won = 0
            agent_matches_won = 0
            agent_hit = 0
            agent_ready = 1320

            if c == 3:
                if trial >= random_trial:
                    dqn_agent.epsilon = 1
                else:
                    dqn_agent.epsilon = dqn_agent.epsilon_min

            while True:
                env.render()
            
                action = dqn_agent.act(cur_state)
                action_array = actions[action]
            
                frame_count = 0
                action_reward = 0

                agent_directional_active = 0
                agent_combat_active = 0
                if (0 <= action and action < 6) or action == 38:
                    agent_directional_active = 1
                else: 
                    agent_combat_active = 1

                invalid_action = False
                damage_dealt = 0
                damage_received = 0
                while True:
                    env.render()
                    frame_count += 1
    
                    pixel_new_state, reward, done, info = env.step(action_array)
                    new_state = dqn_agent.pre_process(info)
                
                    agent_combat_active = new_state[12]
                    agent_jump = new_state[6]

                    new_clock = info["clock"]
                    new_agent_health = info["agent_health"]
                    new_enemy_health = info["enemy_health"]
                    agent_stun = info["agent_stun"]
                    continuetimer = info["continuetimer"]
                    agent_rounds_won = info["agent_rounds_won"]
                    agent_matches_won = info["agent_matches_won"]
                    agent_hit = info["agent_hit"]

                    all_zeros = not np.any(pixel_new_state)
                    if all_zeros:
                        agent_health = 176
                        enemy_health = 176
                        new_agent_health = 176
                        new_enemy_health = 176

                    if info["agent_health"] <= 0 or info["enemy_health"] <= 0 or info["clock"] == 0 or info["clock"] == 5500000000000099 or (not agent_hit == agent_ready and not agent_combat_active) or (agent_jump and agent_directional_active):
                        invalid_action = True
                        break
                    
                    if agent_directional_active and frame_count >= directional_frame_window:
                        break
                    elif not agent_combat_active:
                        break
            
                if states_memory_n > 0:
                    single_new_state = new_state
                    for i in range(len(states_memory)):
                        new_state = new_state + states_memory[i]
                    new_state = new_state + actions_memory
                    states_memory.pop(0)
                    states_memory.append(single_new_state)
                    actions_memory.pop(0)
                    actions_memory.append(action/len(actions))
                new_state = (np.array(new_state)).reshape(1, state_n)

                if frame_count > 1:
                    action_reward = new_agent_health - new_enemy_health
                    if new_agent_health < agent_health and not (agent_health == 176 and new_agent_health == 0):
                        damage_received = agent_health - new_agent_health
                        agent_health = new_agent_health
                    else:
                        damage_received = 0
                    
                    if new_enemy_health < enemy_health and not (enemy_health == 176 and new_enemy_health == 0):
                        damage_dealt = enemy_health - new_enemy_health
                        enemy_health = new_enemy_health

                if not invalid_action and not agent_stun > 0 and not all_zeros and action_step >= actions_memory_n:
                    if frame_count > 1:
                        if not c == 3:
                            dqn_agent.remember(cur_state, action, action_reward, new_state, done)
                            dqn_agent.replay()
                            dqn_agent.target_train()

                        if not agent_directional_active:
                            delay_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
                            _, _, done, _ = env.step(delay_action)
                    done = True if action_step >= max_actions else done

                cur_state = new_state
                action_step += 1

                total_reward += (damage_dealt - damage_received)
                if done:
                    break

                # restrict matches to one character
                if not c == 3:
                    if (agent_matches_won - c) > 0:
                        break
                
            print("Trial ", trial, "matches won = ", agent_matches_won,  " rounds won = ", agent_rounds_won, ", total_reward = ", total_reward, " (epsilon value = ", dqn_agent.epsilon, ")")
            if not c == 3:
                dqn_agent.epsilon *= dqn_agent.epsilon_decay
                dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon)
            else:
                if trial < random_trial:
                    trial_n.append(trial + 1)
                    total_rewards.append(total_reward)
                    rounds_won.append(agent_rounds_won)
                    matches_won.append(agent_matches_won)
                elif trial >= random_trial:
                    random_trial_n.append((trial - random_trial) + 1)
                    random_total_rewards.append(total_reward)
                    random_rounds_won.append(agent_rounds_won)
                    random_matches_won.append(agent_matches_won )
        
        env.close()
        if c == 3:
            plot(trial_n, total_rewards, random_trial_n, random_total_rewards, "Arcade")

            average_rounds_won = sum(rounds_won)/(len(rounds_won)*2)
            average_matches_won = sum(matches_won)/len(matches_won)
            average_won.append([average_rounds_won, average_matches_won])

            random_average_rounds_won = sum(random_rounds_won)/(len(random_rounds_won)*2)
            random_average_matches_won = sum(random_matches_won)/len(random_matches_won)
            random_average_won.append([random_average_rounds_won, random_average_matches_won])

            print("Arcade: average_rounds_won = ", average_rounds_won, ", average_matches_won = ", average_matches_won, ", random_average_rounds_won = ", random_average_rounds_won, ", random_average_matches_won = ", random_average_matches_won)
            print("-----------------------------")

    # each sub-list element is the agent's rounds/matches won for a character
    print("agent's average round/match wins in arcade = ", average_won)

    # each sub-list element is the random's rounds/matches won for a character
    print("random's average round/match wins in arcade = ", random_average_won)

def plot(trial_n, total_rewards, random_trial_n, random_total_rewards, character_name):
    fig = plt.figure()

    slope, intercept, r_value, p_value, std_err = stats.linregress(trial_n, total_rewards)
    line = slope*np.array(trial_n)+intercept
    plt.plot(trial_n, total_rewards, 'bs', trial_n, line)

    random_slope, random_intercept, random_r_value, random_p_value, random_std_err = stats.linregress(random_trial_n, random_total_rewards)
    random_line = random_slope*np.array(random_trial_n)+random_intercept
    plt.plot(random_trial_n, random_total_rewards, 'g^', random_trial_n, random_line)

    name = "saved_plots/" + character_name + ".png"
    fig.savefig(name)

if __name__ == "__main__":
    train_network()

    # show_network()