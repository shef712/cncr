import gym
import retro

gym.logger.set_level(40)

def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env.reset()

    # execute one heavy kick, then two using combat active
    # for the first 40 frames we will wait, wait_frame = 40
    # then we will use the action array to do a heavy kick
    # then we will have a cycle that will repeat the action kick, until finish using combat_active and wait 40 frames again
    # we will do this until time is 1000

    empty_action = [0,0,0,0, 0,0,0,0, 0,0,0,0]
    h_kick_action = [0,0,0,0, 0,0,0,0, 1,0,0,0]
    
    min_frame_window = 1

    frame_count = 1
    action_step_n = 5
    for step in range(action_step_n):
        frame_count = 0
        agent_active = 1

        if step % 2 == 0:
            action = h_kick_action
        else:
            action = empty_action

        print("--- STEP ", step, " ---")

        while agent_active:
            env.render()
            observation, reward, done, info = env.step(action)
            agent_combat_active = info["agent_combat_active"]
            print("agent_combat_active = ", agent_combat_active)

            if action == empty_action and frame_count == min_frame_window - 1:
                agent_active = 0
            elif action == h_kick_action:
                agent_active = agent_combat_active

            frame_count += 1
            
        print("--- STEP ", step, " TOOK ", frame_count, " FRAMES ---")

        
if __name__ == '__main__':
    main()
