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
# a = 0.95
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



# from collections import deque
# memory = deque(maxlen=3)
# memory.append([1, 2, 3, 4])
# memory.append([5, 6, 7, 8])
# memory.append([9, 10, 11, 12])
# memory.append([13, 14, 15, 16])
# for memory_i in range(3):
#     _, b, _, _ = memory[-(memory_i + 1)]
#     # b = memory[-(memory_i + 1)
#     print("b = ", b, " at memory position = ", memory_i)


# a = [1, 2, 3]
# a.pop(0)
# a.append(4)
# print(a)


# a = [[0]*5]*2
# a.append([1,1,1,1,1])
# print(a)




# for i in range(10):
#     print(i)




# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from collections import namedtuple

# n_groups = 3

# agent_wins = (20, 35, 30)
# agent_zero = ()
# for i in range(len(agent_wins)):
#     agent_zero = agent_zero + (0,)

# random_wins = (25, 32, 34)
# random_zero = ()
# for i in range(len(random_wins)):
#     random_zero = random_zero + (0,)

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.35

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# rects1 = ax.bar(index, agent_wins, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=agent_zero, error_kw=error_config,
#                 label='Agent Wins')

# rects2 = ax.bar(index + bar_width, random_wins, bar_width,
#                 alpha=opacity, color='r',
#                 yerr=random_zero, error_kw=error_config,
#                 label='Random Wins')

# ax.set_xlabel('Characters')
# ax.set_ylabel('Matches Won')
# ax.set_title('Comparison between Trained Agent Vs Random Agent')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('Guile', 'Ken', 'Chun-Li'))
# ax.legend()

# fig.tight_layout()
# plt.show()

