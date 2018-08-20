import gym
import retro

gym.logger.set_level(40)

def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env.reset()

    time = 0
    # total_reward = 0
    # reward = 0
    done = False

    while True:
        # action = [B, A, mode, start, up, down, left, right, C, Y, X, Z]
        # B = medium kick, A = light kick, C = heavy kick
        # Y = medium punch, X = light punch, Z = heavy punch
           
        # jump diagonal forward COFIRMED SAME FRAMES AS STRICT VERTICAL JUMP
        action = [0,0,0,0,1,0,0,1,0,0,0,0]

        if time % 100 == 0 and time > 0:
            action = [0,0,0,0,1,0,0,1,0,0,0,0]
        else:
            action = [0,0,0,0,0,0,0,0,0,0,0,0]

        env.render()
        if time % 1 == 0:
            observation, reward, done, info = env.step(action)
            agent_y = info["agent_y"]
            clock = info["clock"]
            print("time = ", time, ", agent_y = ", agent_y)        
        
        
        if time == 500:
            done = True
            
        # total_reward += reward
        if done:
            env.close()
            break
        
        time += 0.25

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
