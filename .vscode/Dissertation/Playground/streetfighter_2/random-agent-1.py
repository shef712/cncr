import gym
import retro

def main():
    # what state is going to load (?) - default state is defined in metadata.json
    # look at the possible arguments for "make()"

    # game = 'StreetFighterIISpecialChampionEdition-Genesis'
    # game = 'Airstriker-Genesis'

    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    # observation = env.reset()
    env.reset()

    time = 0
    total_reward = 0

    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render() # i usally do this before the step function, i wonder if there is a difference (?)
        # - we can stop rendering for certain timesteps
        # not going to render straight away... will it work now (?)
        # seems to be an issue when rendering in openai and nvidia drivers, not really an important problem for now, since i have much more important things to worry to do

        time += 1
        if time % 10 == 0:
            # check if info is not null
            if info:
                info_content = {key: value for key, value in info.items()}
                print('time : ', time, ', info: ', info_content)

            # env.render() # only render every 10 timesteps
        total_reward += reward

        if reward > 0:
            print('time: ', time, ', reward: ', reward, ', current_reward:', total_reward)
        if reward < 0:
            print('time: ', time, ', penalty: ', reward, ', current_reward:', total_reward)

        if done:
            observation = env.reset()
            # This happens both, when time and lives are up, and when game is completed
            # env.render()
            print('done!')
            env.close()
            break

    print('time: ', time, ', total_reward:', total_reward)

if __name__ == '__main__':
    gym.logger.set_level(40)

    main()

    # how does the game know how to restart the game, and even start a match?
    # it must be related to states for the game already included LOOK INTO
    # answered in Learning/streetfighter_2/notes.txt