################################
# EvoMan FrameWork - V1.0 2016 #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
import numpy as np

experiment_name = 'pep_spec'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  playermode='ai',
                  player_controller=player_controller(n_hidden),
                  level=2)

# individual gain = player_energy - enemy_energy
sol = np.loadtxt('solutions_demo/demo_all.txt')

# tests saved demo solutions for each enemy
for en in range(1, 3):
	
	#Update the enemy
	env.update_parameter('enemies',[en])

	env.play(sol)


# env.play()

