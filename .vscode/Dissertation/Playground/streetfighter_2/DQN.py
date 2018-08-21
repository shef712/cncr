import gym
import retro
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

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

intial_state = [0.373134328, 1, 0.626865672, 1, 0, 0, 0, 0]

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
        model.add(Dense(24, input_dim=state_n, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(actions[:6])))
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
            return random.randint(0,len(actions[:6])-1)
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
            if (len(self.memory) == 0):
                agent_x = intial_state[0]
            else:
                agent_x = self.memory[-1][0][0][0]
                memory_agent_x = True
        if enemy_x < min_x or enemy_x > max_x:
            if (len(self.memory) == 0):
                enemy_x = intial_state[2]
            else:
                agent_x = self.memory[-1][0][0][2]
                memory_enemy_x = True
        if agent_health < 0:
            agent_health = min_health
        if enemy_health < 0:
            enemy_health = min_health
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

        # return state
        return [agent_x, agent_y, enemy_x, enemy_y, agent_crouch, enemy_crouch, agent_combat_active, enemy_combat_active]
    
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Versus.RyuVsKen')

    trials  = 5 # 1000
    trial_len = 50 # 500

    min_frame_window = 4
    max_frame_window = 40

    dqn_agent = DQN(env=env)
    for trial in range(trials):
        cur_state = env.reset()
        cur_state = (np.array(intial_state)).reshape(1, 8)

        # start off a new trial with a ready agent
        agent_active = 0

        for action_step in range(trial_len):
            # env.render()

            # for each new action we want to step through, for each action_step, the agent will be ready for a new action because "agent_active" will be 0, so no need ot reset it when it will be reset naturally
            # for when we want to do a new action step, we get the new action from our network
            # we should check if it is directionl or combat, assume combat for now
            # we will need to step through one observation to get the value agent_combat_active to be true, or we can just set it ourselves because we know that a new action has just been given (and a directional would have to have at least four frames anyway which we can count)
            # we will step through the first frame with the new action, and get the reward for said reaction, and continue to step with the same action array, until agent_combat_active is false, or the min_frame_window as passed for direcitonal button
            # in which case we will have accumulated the reward (the reward is given over one frame while agent active anyway so it wont be a crazy high reward anyway)
            # once the agent is not active anymore, we will escape the while loop, train and network with action and reward and the last state, which will essentially treat the last x frames as one state (hopefully any moves that move the player far away and change the enemy x won't have a big impact...)

            action = dqn_agent.act(cur_state)
            action_array = actions[action]
            agent_active = 1

            agent_directional_active = 0
            agent_combat_active = 1
            if (0 < action and action < 5) or action == 12:
                agent_directional_active = 1
            else: 
                agent_combat_active = 1

            frame_count = 1
            total_action_reward = 0

            # action selected = heavy kick
            # so agent is now made active
            # if it takes 17 frames for the heavy kick to execute
            # we can repeat the action for 17 frames, or even safer do the empty action, because i am worried that the heavy kick might repeat, and as long as we pass in the kick action into the network, then the empty action for one frame won't affect it
            # still, if we executing the heavy kick for 17 frames, it will be made non-active in the result frame, so new_state will have agent_comabt_active = 0
            # so we should not need to do all that above, okay

            while agent_active:
                new_state, reward, done, info = env.step(action_array)
                new_state = (np.array(dqn_agent.pre_process(info))).reshape(1,8)
                agent_combat_active = new_state[6]

                if agent_directional_active:
                    framecount += 1
                    if frame_count > min_frame_window:
                        # no more frames to repeat directional action now
                        agent_active = 0
                elif not agent_combat_active:
                    # no more frames to wait for combat action to repeat now
                    agent_active = 0
                
                # we still care about the reward at the frame where agent is nonactive, because it moved from active to non-active, and the move has just finsihed
                total_action_reward += reward

            # we care about the state of the env as soon as the agent executed the move, for example, if the agent had just began to do a hadouken, and the player jumped over it and the enemy is close enough to jump and attack, all while the hadouken is firing and our agent can't attack then 

            # what happens when we do try and do actions while hit in some long ass combo, wont the actions be given penalties because of being hit! will only let score be the reward for, in that case, just to get some immediate results

            # we also care about the resulting state of said action
            # so basically we are skipping the states of the frames it takes to execute an action, with the minimum being 4 frames, though a combat action could take less than 4... well leave directional as 4 anyway

            dqn_agent.remember(cur_state, action, total_action_reward, new_state, done)
            dqn_agent.replay()
            dqn_agent.target_train()

            cur_state = new_state
            if done:
                break
        
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if trial % 10 == 0:
                print("Saving failed trial model")
                dqn_agent.save_model("saved_models/trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("saved_models/success.model")
            break

if __name__ == "__main__":
    main()