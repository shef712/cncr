import gym
import retro

gym.logger.set_level(40)
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Versus.RyuVsKen')
    env.reset()

    time = 0
    total_reward = 0
    reward = 0
    done = False

    current_action = 0
    button_action = 0

    repeat_action = 12
    # looks like repeating the button press (punch) for the hadouken for the amount of frames it takes to do it, seems to only fire hadoukens pretty accurately
    # is this something i should let the agent figure out for itself, or something i should plug into the game...
    
    while True:
        # B = medium kick, A = light kick, C = heavy kick
        # Y = medium punch, X = light punch, Z = heavy punch
        # action = [B, A, mode, start, up, down, left, right, C, Y, X, Z]
        # where [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] = low heavy kick
        # action = env.action_space.sample()
        # print ("action = ", action)
        
        # actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]       

        print("current_action = ", current_action)

        actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] 

        # directional presses happen in one frame, but button presses happen over 20... but what if a move is finished before 20, this this button press will be repeating for (20-x) frames... hmm how important is this... well if the player needs to block then it would be!

        action = actions[current_action]
        observation, reward, done, info = env.step(action)
        env.render()

        if current_action == 3:
            if button_action == repeat_action:
                button_action = 0
                current_action = -1
            else:
                current_action = 2
                button_action += 1
            
        # we need a set amount of frames for each action, so 1/3 of a second, so 20 frames should do it

        # we repeat an action 

        # if repeat_action < 60:
        #     # 
        #     action = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        
        # if repeat_action >= 60:
        #     action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # if next_action == 120:
        #     repeat_action = 0
        #     next_action = 0

        # testing with "done" was a bit silly, because down needs to be held for a few frames to be able to go all the way down
        # how many frames then? use 20 to repeat actions and see the result, start with the "down" press
        # does press down every frame count as holding down? i hope so... YES it does, thankfully, i guess the game must have some way to convert holding a button press to the button press for every frame

        # print ("action = ", action)
        # print ("repeat_action = ", repeat_action, ", next_action = ", next_action)

        # observation, reward, done, info = env.step(action)
        # env.render()

        if time == 4200:
            done = True
            
        total_reward += reward
        # if reward > 0:
        #     print('time: ', time, ', reward: ', reward, ', current_reward:', total_reward)
        # if reward < 0:
        #     print('time: ', time, ', penalty: ', reward, ', current_reward:', total_reward)
        if done:
            print('done!')
            env.close()
            break
        
        time += 1
        current_action += 1

    print('time: ', time, ', total_reward:', total_reward)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
