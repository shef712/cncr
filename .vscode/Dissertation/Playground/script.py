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

# import matplotlib.pyplot as plt

# a = 0.997
# b = 1
# count = 1
# x = []
# x.append(count)
# y = []
# y.append(b)
# while True:
#     b = b * a
#     count += 1

#     x.append(count)
#     y.append(b)

#     # print("b = ", b, " COUNT ", count)
#     if b <= 0.1:
#         break

# plt.plot(x, y)
# plt.show()



from collections import deque
memory = deque(maxlen=3)
memory.append([1, 2, 3, 4])
memory.append([5, 6, 7, 8])
memory.append([9, 10, 11, 12])
memory.append([13, 14, 15, 16])
for memory_i in range(3):
    _, b, _, _ = memory[-(memory_i + 1)]
    # b = memory[-(memory_i + 1)
    print("b = ", b, " at memory position = ", memory_i)