################################
# EvoMan FrameWork - V1.0 2016 #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
import numpy as np
import time

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
# sol = np.loadtxt('solutions_demo/demo_all.txt')

n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

pop_size = 4
max_generations = 3
print(f"Vars are {n_vars}.")

start_time = time.time()

def simulation(env,x):
    f,*_ = env.play(pcont=x)
    # print(f"simulation {f},{p},{e},{t}")
    return f

def evaluate(x):
    return np.array([simulation(env,y) for y in x])

def population_reduction(pop):
    c1 = np.random.randint(0,pop.shape[0],1)
    c2 = np.random.randint(0,pop.shape[0],1)
    # print("c1, c2", c1,c2, "population shape", pop.shape[0])
    # print(pop[c1][0].shape, pop[c1].shape )
    return pop[c1][0] if fitness_pop[c1] > fitness_pop[c2] else pop[c2][0] 

def crossover(population):
    print("inside")
    print(np.zeros((0,n_vars)).shape)
    print("Pop shape", population.shape[0])
    for p in range(0,population.shape[0],2):
        p1 = population_reduction(population)
        p2 = population_reduction(population)
    return 
# init population randomly

if not os.path.exists(experiment_name+'/solution'):
    print("Creating new population")
    population = np.random.uniform(-1, 1, (pop_size, n_vars))
    fitness_pop = evaluate(population)
    # the x amount of best values
    best_in_pop = np.argsort(fitness_pop)[-2:]
    mean, std = np.mean(fitness_pop), np.std(fitness_pop)
    generation_number = 1
    env.update_solutions([population, fitness_pop])
else:
    # TODO check if this works well 
    print("Using old population")
    env.load_state()
    population, fitness_pop = env.solutions[:2]
    best_in_pop = np.argsort(fitness_pop)[-2:]
    mean, std = np.mean(fitness_pop), np.std(fitness_pop)
    file_gen = open(experiment_name+'/gen_number.txt', 'r')
    generation_number = int(file_gen.readline())
    file_gen.close()

# saves the results for population
text_file = open(experiment_name+"/pep_results.txt", "a")
n = text_file.write(f'\n\nBest:{fitness_pop[best_in_pop[-1]]} \nSecond best:{fitness_pop[best_in_pop[-2]]}\nMean:{mean}\nstd:{std} ')
text_file.close()

# apply evolution
best_previous, second_best_previous = fitness_pop[best_in_pop[-1]], fitness_pop[best_in_pop[-2]]

for i in range(generation_number, max_generations+1):
    kids = crossover(population)


print(f"Population shape: {population.shape}")
print(f"Fitness pop: {fitness_pop}")
print(f"Best: {best_in_pop[-1]}, Second best: {best_in_pop[-2]}")


