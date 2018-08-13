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
    # looks like repeating the button press (punch) for the hadouken for the amount of frames it needs, seems to fire hadoukens pretty accurately
    # is this something i should let the agent figure out for itself, or something i should plug into the game manually?
    
    while True:
        # B = medium kick, A = light kick, C = heavy kick
        # Y = medium punch, X = light punch, Z = heavy punch
        # action = [B, A, mode, start, up, down, left, right, C, Y, X, Z]
        # where [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] = low heavy kick
        
        actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]       

        # directional presses happen in one frame, but button presses continue to execute over many frames
        # and it looks like, open ai will overwrite the action if another action is applied in a following frame
        # so the agent needs to learn to repeat the button press action for x frames, to get the reward of of the combat button pressed
        # hopefully it will realise that repeating for x frames gets most reward, though it will need to go through that branch of repeating the button, which can be enhanced if there is no reward for no health change and a massive reward for decreasing enemy's health
        # e.g. will it be able to repeatedly choose the punch button (assuming hadouken) when there is only a reward after 13 frames when the enemy is hit?
        # since it ranks each action based on expected reward, a kick may do better in some scenarios of starting a hadouken move, and then that move is lost forever, though that would mean the kick is better for that state i guess
        # will it be able to realise repeatedly pressing punch 13 times will give it the best reward, should be able to, since all other moves will give 0 for the next 13 anyway
        # and assuming that the character does not need to move to avoid losing health, this is the best option to inflict damage, assuming the enemy isnt close enough to jump over and attack either

        # since we will be storing all memory anyway, we could provide 2 extra inputs that make up the state for previous_move_id and move_repeated_n
        

        # wait, we need to alter the action space into the 35 possible "paired" actions anyway, would this make a difference for above? if we know what action combo we are doing, we could just repeat for that? though this would need to be repeated for x frames still to see the paired action execute

        # its possible that for each paired action, the associated combat button could have a fixed number of frames, preferably fixed for the combat button across all paired actions

        # though it is likley that each move and combo result will need different amount of frames and it is likely we need different amounts for each player too... so thats something we will just have to rely on the network to discover then...

        # action = actions[current_action]
        observation, reward, done, info = env.step(action)
        env.render()

        if current_action == 3:
            if button_action == repeat_action:
                button_action = 0
                current_action = -1
            else:
                current_action = 2
                button_action += 1
        
        # hopefully, if the agent is in states where it needs to block (so if positions_delta is low), then it continually presses the opposite directional button or the down button along with it, to block (low)... depending on the states it is in
        # but what would the states for defending and attacking even differ?

        if time == 4200:
            done = True
            
        total_reward += reward
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
