from stable_baselines3 import PPO
from NineDown import NineDown
import os

game = NineDown()
env = game.players[0]

models_dir = 'models/SelfvsSelf_5'
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, device=0)
if not os.path.exists(logdir):
    os.makedirs(logdir)


env.reset()

# # Load most recent model
models = [int(file[:-4]) for file in os.listdir(models_dir)]
models.sort()

model = PPO.load(f'{models_dir}/{models[-1]}.zip', env, verbose=1, device=0)


TIMESTEPS = 200_000
i = models[-1]/TIMESTEPS+1
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=os.path.split(models_dir)[-1])
    model.save(f'{models_dir}/{int(i*TIMESTEPS)}')
    i += 1
