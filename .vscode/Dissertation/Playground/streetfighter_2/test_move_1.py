import gym
import retro

gym.logger.set_level(40)

def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Versus.RyuVsKen')
    env.reset()

    time = 0
    # total_reward = 0
    # reward = 0
    done = False

    # current_action = 0
    # button_action = 0

    # repeat_action = 12
    # looks like repeating the button press (punch) for the hadouken for the amount of frames it needs, seems to fire hadoukens pretty accurately
    # is this something i should let the agent figure out for itself, or something i should plug into the game manually?
    
    # hadouken did well with max frame wait of 6

    #frame_wait = 18
    frame_wait = 20
    start_frame = 80

    wait_time = 80
    repeated_time = 0

    agent_combat_active = 0

    while True:
        # B = medium kick, A = light kick, C = heavy kick
        # Y = medium punch, X = light punch, Z = heavy punch
        # action = [B, A, mode, start, up, down, left, right, C, Y, X, Z]
        # where [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] = low heavy kick
        
        # hadouken
        # actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]   
        # actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]]   
        # shoryuken    
        # actions = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]    
        
        # jump diagonal forward
        # action = [0,0,0,0,1,0,0,1,0,0,0,0]

        # jab
        # action = [0,0,0,0, 0,0,0,0, 0,0,1,0]
        
        # ---

        # its possible that for each paired action, the associated combat button could have a fixed number of frames, preferably fixed for the combat button across all paired actions

        # though it is likley that each move and combo result will need different amount of frames and it is likely we need different amounts for each player too... so thats something we will just have to rely on the network to discover then...

        # ---

        # hopefully, if the agent is in states where it needs to block (so if positions_delta is low), then it continually presses the opposite directional button or the down button along with it, to block (low)... depending on the states it is in
        # but what would the states for defending and attacking even differ?
        # well without knowing the enemy's move id, the agent probably wont block, but this shouldnt mess training if the reward function doesnt give penalities for being hit

        # ---

        # action = actions[current_action]

        # if current_action == 3:
        #     if button_action == repeat_action:
        #         button_action = 0
        #         current_action = -1
        #     else:
        #         current_action = 2
        #         button_action += 1

        # if time == 4200:
        #     done = True

        if time % 1 == 0:
            # if time <= start_frame:
            #     action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # elif time > start_frame and time <= (start_frame + frame_wait):
            #     action = actions[0]
            # elif time > (start_frame + frame_wait) and time <= (start_frame + (frame_wait*2)):
            #     action = actions[1]
            # elif time > (start_frame + (frame_wait*2)) and time <= (start_frame + (frame_wait*3)):
            #     action =  actions[2]
            # elif time > (start_frame + (frame_wait*3)) and time <= (start_frame + (frame_wait*4)):
            #     action =  actions[3]
            # elif time > (start_frame + (frame_wait*4)):
            #     action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #     time = -1*wait_time
            #     repeated_time += 1
            
            if time <= start_frame:
                action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
            elif time > start_frame and time <= start_frame*2:
                action = [0,0,0,0, 0,0,0,0, 1,0,0,0]
            elif time > start_frame*2:
                repeated_time += 1

            observation, reward, done, info = env.step(action)
            print ("time = ", time, ", agent_combat_active = ", info["agent_combat_active"])

        env.render()

        if repeated_time == 2:
            done = True

        #total_reward += reward
        if done:
            env.close()
            break
        
        time += 0.25 # lowest value we can increment, so it looks like we need to step through at least a minimum of every 4 frames...
        # current_action += 1

    #print('time: ', time, ', total_reward:', total_reward)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
