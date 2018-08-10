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
        action = env.action_space.sample()
        # print ("action = ", action)
        
        # actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]       

        print("action = ", action)

        # actions = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] 

        # directional presses happen in one frame, but button presses happen over 20... but what if a move is finished before 20, this this button press will be repeating for (20-x) frames... hmm how important is this... well if the player needs to block then it would be!
        # so the agent will learn how long it should repeat the button press action, because if it has pressed it for so many frames, it will hopefully learn that pressing it continuously will get some reward, although, the reward for no health lost surely cannot be positive, because then if it doesn't complete the combo, then it will not do anything...
        # will it be able to repeatedly choose the punch button (assuming hadouken) when there is only a reward after 13 frames when the enemy is hit?
        # i mean, assuming it ranks each action based on expected reward, a kick may do better in some scenarios of starting a hadouken move, and then that move is lost forever, though that would mean the kick is better for that state i guess
        # the point is, will it be able to realise repeatedly pressing punch 13 times will give it the best reward, should be able to, since all other moves will give 0 for the next 13 anyway
        # and assuming that the character does not need to move to avoid losing health, this is the best option to inflict damage, assuming the enemy isnt close enough to jump over and attack either
        # ALSO i will have to give the last 30 (provisionary) frames as an input, maybe just move id's from these states, but how would i even go about create a NN with these inputs! this is most important for now, before i can even hope that this works 
        # i mean how would that work... i kinda get how one state would work, which i should probably focus on after reading the q-network article, but adding in the past states in the inputs... is it as simple as adding in the states 30 times for each timestep as inputs?

        # well how will the previous state be distinguished from the current state? i mean the previous states will have actions defined, whereas this state wont, so maybe just the actions will be used, i dunno, somehow i need to feed the previous 30 actions as an input to help decide the current state to continue the move or not
        

        # wait, we need to alter the action space into the 35 action set, would this make a difference for above? if we know what action combo we are doing, we could just repeat for that?

        # can i just repeat the button presses for 20 frames, or whatever each button needs? maybe a,b,c,x,y,z all need 13 frames repeated too... TEST
        # if it looks like different amounts for each button press, then it is likely we need different amounts for each player, and thats something we will have to rely on the network to discover then... great

        # action = actions[current_action]
        observation, reward, done, info = env.step(action)
        env.render()

        # if current_action == 3:
        #     if button_action == repeat_action:
        #         button_action = 0
        #         current_action = -1
        #     else:
        #         current_action = 2
        #         button_action += 1
            
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
