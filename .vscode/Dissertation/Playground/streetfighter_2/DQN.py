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
    # directions action
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],

    # combat action
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    
    # empty action
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

normalised_empty_action = 12/len(actions)
intial_state = [0.373134328,1,0.626865672,1, 176,176, 0,0,0,0, 0.253731344,1, 0,0, normalised_empty_action,normalised_empty_action,normalised_empty_action]

class DQN:
    def __init__(self, env):
        self.env = env

        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_n = len(intial_state)

        model.add(Dense(9, input_dim=state_n, activation="relu"))

        # model.add(Dense(24, input_dim=state_n, activation="relu"))
        # model.add(Dense(48, activation="relu"))
        # model.add(Dense(24, activation="relu"))

        # model.add(Dense(len(actions[:6])))

        model.add(Dense(len(actions)))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32

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
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # return random.randint(0,len(actions[:6])-1)
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
        min_y = 122 # peak of jump
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
        
        memory_agent_x = False
        memory_enemy_x = False
        # clean up
        if agent_x < min_x or agent_x > max_x:
            memory_agent_x = True
            if (len(self.memory) == 0):
                agent_x = intial_state[0]
            else:
                agent_x = self.memory[-1][0][0][0]
        if enemy_x < min_x or enemy_x > max_x:
            memory_enemy_x = True
            if (len(self.memory) == 0):
                enemy_x = intial_state[2]
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
        agent_x = (agent_x - min_x) / range_x if not memory_agent_x else agent_x
        agent_y = (agent_y - min_y) / range_y
        enemy_x = (enemy_x - min_x) / range_x if not memory_enemy_x else agent_x
        enemy_y = (enemy_y - min_y) / range_y
        agent_health = agent_health / max_health
        enemy_health = enemy_health / max_health
        clock = clock / max_clock

        absolute_diff = math.sqrt( ((agent_x - enemy_x)**2) + ((agent_y - enemy_y)**2) )

        return [agent_x, agent_y, enemy_x, enemy_y, 
        agent_health, enemy_health, 
        agent_jump, enemy_jump, agent_crouch, enemy_crouch, 
        absolute_diff, clock, 
        agent_combat_active, enemy_combat_active, 
        0, 0, 0]
    
def train_network():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.Ryu')

    trials  = 100 # 10000

    min_frame_window = 20
    max_frame_window = 40
    
    max_actions = 10000
    penalty = -5000

    total_rewards = []

    dqn_agent = DQN(env=env)
    for trial in range(trials):
        cur_state = env.reset()
        cur_state = (np.array(intial_state)).reshape(1, 17)

        actions_memory = [normalised_empty_action, normalised_empty_action, normalised_empty_action]

        cur_agent_health = 176

        total_reward = 0
        action_step = 0
        rounds_won = 0
        done = False
        while True:
            env.render()

            action = dqn_agent.act(cur_state)
            action_array = actions[action]
            
            agent_active = 1
            frame_count = 0
            total_action_reward = 0

            agent_directional_active = 0
            agent_combat_active = 0
            if (0 <= action and action < 6) or action == 12:
                agent_directional_active = 1
            else: 
                agent_combat_active = 1

            new_clock = 0
            while agent_active:
                env.render()
                frame_count += 1

                new_state, reward, done, info = env.step(action_array)
                new_state = (np.array(dqn_agent.pre_process(info))).reshape(1,17)
                agent_combat_active = new_state[0][6]
                new_clock = info["clock"]
                new_agent_health = info["agent_health"]
                rounds_won = info["agent_matches_won"]
                done = info["continuetimer"] > 0
                if done:
                    break

                total_action_reward += reward
                # if not reward == 0:
                    # print("reward = ", reward)

                if agent_directional_active:
                    if frame_count == min_frame_window:
                        agent_active = 0
                elif not agent_combat_active:
                    agent_active = 0

            for taken_action in range(3, 0, -1):
                new_state[0][-taken_action] = actions_memory[-taken_action]
            actions_memory.append(action/len(actions))
            
            if cur_agent_health > new_agent_health:
                # print("PENALTY! Agent health is now:", new_agent_health)
                cur_agent_health = new_agent_health
                total_action_reward = total_action_reward - penalty

            # print("--- STEP ", action_step, "(", action_array , "), TOOK ", frame_count, " FRAMES, REWARD = ", total_action_reward, " ---")
            
            if new_clock > 0 and (agent_directional_active or agent_combat_active) and not done:
                print("TRAINING at --- STEP ", action_step)
                dqn_agent.remember(cur_state, action, total_action_reward, new_state, done)
                dqn_agent.replay()
                dqn_agent.target_train()

            if not done:
                delay_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
                _, _, done, _ = env.step(delay_action)
                
            print("done = ", done)
            done = True if action_step >= max_actions else done

            if done:
                break
            
            cur_state = new_state
            action_step += 1
            total_reward += total_action_reward

        print("Trial ", trial, ", rounds won = ", rounds_won, ", total_reward = ", total_reward)
        total_rewards.append(total_reward)
        if trial % 100 == 0:
            dqn_agent.save_model("saved_models/ryu-trial-{}.model".format(trial))

    dqn_agent.save_model("saved_models/ryu.model")
    plot_running_avg(total_rewards)

def show_network():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.Ryu')

    dqn_agent = DQN(env=env)
    dqn_agent.load_model("saved_models/ryu-trial-600.model")

    trials  = 1

    min_frame_window = 10
    max_frame_window = 40

    max_actions = 10000

    dqn_agent = DQN(env=env)
    for trial in range(trials):
        cur_state = env.reset()
        cur_state = (np.array(intial_state)).reshape(1, 17)

        actions_memory = [normalised_empty_action, normalised_empty_action, normalised_empty_action]

        action_step = 0
        while True:
            env.render()

            action = dqn_agent.act(cur_state)
            action_array = actions[action]
            
            agent_active = 1
            frame_count = 0
            total_action_reward = 0

            agent_directional_active = 0
            agent_combat_active = 0
            if (0 <= action and action < 6) or action == 12:
                agent_directional_active = 1
            else: 
                agent_combat_active = 1

            while agent_active:
                env.render()
                frame_count += 1

                new_state, reward, done, info = env.step(action_array)
                new_state = (np.array(dqn_agent.pre_process(info))).reshape(1,17)
                agent_combat_active = info["agent_combat_active"]

                total_action_reward += reward

                if agent_directional_active:
                    if frame_count == min_frame_window:
                        agent_active = 0
                elif not agent_combat_active:
                    agent_active = 0

            for taken_action in range(3, 0, -1):
                new_state[0][-taken_action] = actions_memory[-taken_action]
            actions_memory.append(action/len(actions))

            # print("--- STEP ", action_step, " TOOK ", frame_count, " FRAMES, REWARD = ", total_action_reward, " ---")

            done = True if action_step >= max_actions else done

            if done:
                break
            else:
                delay_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
                env.step(delay_action)
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