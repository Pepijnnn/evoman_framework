#https://neat-python.readthedocs.io/en/latest/installation.html	
#https://neat-python.readthedocs.io/en/latest/customization.html <-- to add activation functions
import neat

import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
#from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import pickle as pkl

experiment_name = 'TASK1_NEAT_TEST_2'
config_filename_save = "NEAT_TEST1_CONFIG1"
run_mode = 'train'
n_hidden_neurons = 10 #,-- mandotoro
n_pop = 20
gens = 10

if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

class player_controller(Controller):
	def control(self, params, cont):
		net = cont
		output = net.activate(params)
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
				  enemies=[2],
				  playermode="ai",
				  player_controller=player_controller(),
				  enemymode="static",
				  level=2,
				  speed="fastest")
# tried fitness functions
# f2 = 2**((1001 - t)/1000) * 10**((p-e)/100)

# runs simulation
def simulation(env,x, test=False):
	f,p,e,t = env.play(pcont=x)
	# f2 = 2**((1001 - t)/1000) * 10**((p-e)/100)
	if test == True:
		indiv_gain = p - e
		return (f, indiv_gain)
	print("Old cool fitness score: {}".format(f))
	return f

def evaluate(x, test=False):
	return np.array([simulation(env,y,test) for y in x])

def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		genome.fitness = 100.0
		net = neat.nn.RecurrentNetwork.create(genome, config)
		f = simulation(env, net)
		genome.fitness = (f)

def run():
	config_path = 'neat_config.txt'
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation,
							config_path)

	if run_mode =='test':
		for i in range(5):
			# bsol = np.loadtxt(experiment_name+'/best.txt')
			with open(f'{experiment_name}/winner.pkl', 'rb') as f:
				winner = pkl.load(f)
			winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
			print( '\n RUNNING SAVED BEST SOLUTION \n')
			env.update_parameter('speed','normal')
			# fpop_ig_array = evaluate([bsol], True)
			farr = simulation(env, winner_net, True)
			print(f"Fitness : {farr[0]}\nIndividual Gain: {farr[1]}")
			file_best  = open(experiment_name+'/best_ig.txt','a')
			file_best.write('instance individual_gain') if i == 0 else None
			file_best.write(f'\n {i} {farr[1]}')
			file_best.close()
		sys.exit(0)

	config.pop_size = n_pop
	pop = neat.Population(config)

	pop.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)

	checkpoint = neat.Checkpointer()
	pop.add_reporter(checkpoint)

	generations = gens
	winner = pop.run(eval_genomes, generations)


	print('\nBest genome:\n{!s}'.format(winner))
	with open(f"{experiment_name}/winner.pkl", "wb") as f:
		pkl.dump(winner, f)
		f.close()

	checkpoint.save_checkpoint(config, pop,neat.DefaultSpeciesSet, 1 )


if __name__ == '__main__':
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward')
	
	
	run()