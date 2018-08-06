# Testing gym

# # import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

# WORKS

# Testing retro-gym

# import retro
# def main():
#     env = retro.make(game='Airstriker-Genesis', state='Level1')
#     obs = env.reset()
#     while True:
#         obs, rew, done, info = env.step(env.action_space.sample())
#         env.render()
#         if done:
#             obs = env.reset()
# if __name__ == '__main__':
#     main()

# WORKS
# Can get a "Universe" set of games now from various console systems

