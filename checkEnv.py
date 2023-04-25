# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:51:14 2023

@author: braxt
"""

from stable_baselines3.common.env_checker import check_env
from NineDown import NineDown, Player


game = NineDown()
env = game.players[0]
# It will check your custom environment and output additional warnings if needed
check_env(env)
