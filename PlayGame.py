from stable_baselines3 import PPO
from NineDown import NineDown
import cv2
import os

game = NineDown()
env = game.players[0]

models_dir = 'models/SelfvsSelf_5'
logdir = 'logs'

# # Load most recent model
models = [int(file[:-4]) for file in os.listdir(models_dir)]
models.sort()

model = PPO.load(f'{models_dir}/{models[-1]}.zip', env, verbose=1, device=0)

observation = env.reset()
while True:
    # env.render()
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, _ = env.step(action, realPlayer=1)
    if done:
        env.render()
        scores = {}
        for player in env.ParentGame.players:
            # Flip each unflipped card over to reveal its value
            scores[player.id] = player.get_score()
        scores = sorted(scores.items(), key=lambda x: x[1])
        print(f'PLayer ID {scores[0][0]} won with {scores[0][1]} points!!')
        for i, score in enumerate(scores[1:]):
            print(f'Place number {i+2}: Player ID {score[0]} with {score[1]}')
        print(f'Reward: {reward}')
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
