import gym
import retro
import numpy as np
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt 
import math 

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
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]

actions_memory_n = 3
state_n = (14 + actions_memory_n)

class DQN:
    def __init__(self, env):
        self.env = env

        self.memory = deque(maxlen=2000)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997 # 0.97, 0.997

        self.learning_rate = 0.015 # 0.005
        self.tau = 0.2 # 0.125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(42, input_dim=state_n, activation="relu"))
        model.add(Dense(len(actions)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        # batch_size = 32
        batch_size = 24

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

        for n in range(actions_memory_n):
            state.append(0)

        return state
    
def train_network():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.Ryu')
    dqn_agent = DQN(env=env)

    trials = 1500 # 150, 1500
    min_frame_window = 4
    max_actions = 20000

    total_rewards = []

    agent_health = 176
    enemy_health = 176

    trial_n = []
    epsilon_n = []
    for trial in range(trials):
        cur_state = env.reset()
        all_actions_memory = []
        normalised_empty_action = (len(actions)-1)/len(actions)
        intial_state = [0.373134328,1,0.626865672,1, 1,1, 0,0,0,0, 0.253731344,1, 0,0]
        for n in range(actions_memory_n):
            intial_state.append(normalised_empty_action)
            all_actions_memory.append(normalised_empty_action)
        cur_state = (np.array(intial_state)).reshape(1, state_n)

        action_step = 0
        total_reward = 0
        
        done = False

        new_clock = 0
        agent_stun = 0
        continuetimer = 0
        agent_rounds_won = 0
        agent_matches_won = 0
        # above is setup for a new trial
        while True:
            # env.render()
            
            action = dqn_agent.act(cur_state)
            action_array = actions[action]
            
            frame_count = 0
            total_action_reward = 0

            agent_directional_active = 0
            agent_combat_active = 0
            if (0 <= action and action < 6) or action == 12:
                agent_directional_active = 1
            else: 
                agent_combat_active = 1

            invalid_action = False
            damage_dealt = 0
            # above is setup for a new action
            while True:
                # env.render()
                frame_count += 1
 
                pixel_new_state, reward, done, info = env.step(action_array)
                new_state = (np.array(dqn_agent.pre_process(info))).reshape(1,state_n)
            
                agent_combat_active = new_state[0][12]

                new_clock = info["clock"]
                new_agent_health = info["agent_health"]
                new_enemy_health = info["enemy_health"]
                agent_stun = info["agent_stun"]
                continuetimer = info["continuetimer"]
                agent_rounds_won = info["agent_rounds_won"]
                agent_matches_won = info["agent_matches_won"]

                all_zeros = not np.any(pixel_new_state)
                if all_zeros:
                    agent_health = 176
                    enemy_health = 176
                    new_agent_health = 176
                    new_enemy_health = 176

                if info["agent_health"] <= 0 or info["enemy_health"] <= 0 or info["clock"] == 0 or info["clock"] == 5500000000000099:
                    invalid_action = True
                    break
                
                if agent_directional_active:
                    if agent_combat_active:
                        agent_directional_active = 1
                    elif frame_count >= min_frame_window:
                        break
                elif not agent_combat_active:
                    break

            for taken_action in range(actions_memory_n, 0, -1):
                new_state[0][-taken_action] = all_actions_memory[-taken_action]
            all_actions_memory.append(action/len(actions))
            
            if frame_count > 1:
                total_action_reward = new_agent_health - new_enemy_health
                
            # print("--- STEP ", action_step, "(", action_array , "), TOOK ", frame_count, " FRAMES, REWARD = ", total_action_reward, " ---")
            
            if not invalid_action and not agent_stun > 0 and not all_zeros:
                if frame_count > 1:
                    # print("TRAINING at --- STEP ", action_step)
                    dqn_agent.remember(cur_state, action, total_action_reward, new_state, done)
                    dqn_agent.replay()
                    dqn_agent.target_train()

                    if not agent_directional_active:
                        delay_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
                        _, _, done, _ = env.step(delay_action)

                done = True if action_step >= max_actions else done

            cur_state = new_state
            action_step += 1

            if (new_enemy_health < enemy_health):
                damage_dealt = enemy_health - new_enemy_health
                enemy_health = new_enemy_health
            else:
                damage_dealt = 0
            total_reward += damage_dealt

            if done:
                break

        dqn_agent.epsilon *= dqn_agent.epsilon_decay
        dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon)
        print("Trial ", trial, "matches won = ", agent_matches_won,  " rounds won = ", agent_rounds_won, ", total_reward = ", total_reward, " (epsilon value = ", dqn_agent.epsilon, ")")

        total_rewards.append(total_reward)
        trial_n.append(trial)
        epsilon_n.append(dqn_agent.epsilon)

        if trial % 75 == 0:
            dqn_agent.save_model("ryu/guile-trial-{}.h5".format(trial))

    #plot_running_avg(total_rewards)
    plt.plot(trial_n, total_rewards)
    avg_reward = sum(total_rewards)/len(total_rewards)
    epsilon_n = [i*avg_reward for i in epsilon_n]
    plt.plot(trial_n, epsilon_n)
    plt.show()

def show_network():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.Ryu')
    dqn_agent = DQN(env=env)
    dqn_agent.load_model("ryu/guile.h5")
    dqn_agent.epsilon = 0

    trials = 1
    min_frame_window = 4
    max_actions = 20000
    for trial in range(trials):
        cur_state = env.reset()
        all_actions_memory = []
        normalised_empty_action = (len(actions)-1)/len(actions)
        intial_state = [0.373134328,1,0.626865672,1, 1,1, 0,0,0,0, 0.253731344,1, 0,0]
        for n in range(actions_memory_n):
            intial_state.append(normalised_empty_action)
            all_actions_memory.append(normalised_empty_action)
        cur_state = (np.array(intial_state)).reshape(1, state_n)

        action_step = 0
        done = False
        while True:
            env.render()

            action = dqn_agent.act(cur_state)
            action_array = actions[action]
            
            frame_count = 0
            total_action_reward = 0

            agent_directional_active = 0
            agent_combat_active = 0
            if (0 <= action and action < 6) or action == 12:
                agent_directional_active = 1
            else: 
                agent_combat_active = 1

            while True:
                env.render()
                frame_count += 1

                new_state, reward, done, info = env.step(action_array)
                new_state = (np.array(dqn_agent.pre_process(info))).reshape(1,state_n)
                
                agent_combat_active = new_state[0][12]

                if agent_directional_active:
                    if agent_combat_active:
                        agent_directional_active = 1
                    elif frame_count >= min_frame_window:
                        break
                elif not agent_combat_active:
                    break

            for taken_action in range(actions_memory_n, 0, -1):
                new_state[0][-taken_action] = all_actions_memory[-taken_action]
            all_actions_memory.append(action/len(actions))

            if frame_count > 1:
                if not agent_directional_active:
                    delay_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
                    _, _, done, _ = env.step(delay_action)

            done = True if action_step >= max_actions else done

            if done:
                break

            cur_state = new_state
            action_step += 1
            
def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

if __name__ == "__main__":
    train_network()

    # show_network()