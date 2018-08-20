import gym
import retro
gym.logger.set_level(40)
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env.reset()

    time = 0
    total_reward = 0
    while True:
        action = env.action_space.sample()
        action_n = env.action_space
        observation, reward, done, info = env.step(action)
        env.render()

        time += 1
        if time % 10 == 0:
            if info:
                info_content = {key: value for key, value in info.items()}
                # print('time : ', time, ', info: ', info_content)
        total_reward += reward

        # if reward > 0:
            # print('time: ', time, ', reward: ', reward, ', current_reward:', total_reward)
        # if reward < 0:
            # print('time: ', time, ', penalty: ', reward, ', current_reward:', total_reward)
        
        info_content = {key: value for key, value in info.items()}
        print("info_content = ", info_content)

        if time == 1:
            done = True
        if done:
            observation = env.reset()
            print('done!')
            env.close()
            break
    print('time: ', time, ', total_reward:', total_reward)

if __name__ == '__main__':
    main()