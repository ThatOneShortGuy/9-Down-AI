from NineDown import NineDown
import numpy as np

game = NineDown()
env = game.players[0]

observation = env.reset()

print(observation)
print(env.action_space)
print(env.observation_space)

observation = np.array([observation['Card in Hand']] + list(observation['Remaining Cards in Deck']) + list(observation['Table']))
print(observation)
print(observation.shape)