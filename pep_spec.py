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

pop_size = 20
# num_pop=4
max_generations = 10
print(f"Vars are {n_vars}.")

start_time = time.time()

def simulation(env,x):
    f,*_ = env.play(pcont=x)
    # print(f"simulation {f},{p},{e},{t}")
    return f

def evaluate(x):
    return np.array([simulation(env,y) for y in x])

def population_random_most_fit_reduction(pop):
    c1 = np.random.randint(0,pop.shape[0],1)
    c2 = np.random.randint(0,pop.shape[0],1)
    # print("c1, c2", c1,c2, "population shape", pop.shape[0])
    # print(pop[c1][0].shape, pop[c1].shape )
    return (pop[c1][0],fitness_pop[c1]) if fitness_pop[c1] > fitness_pop[c2] else (pop[c2][0],fitness_pop[c2])


mutation_factor = 0.1

# for every gen randomly choose to mutate or not
def mutate(child):
    # print("old child",child)
    for i in range(len(child)):
        if np.random.rand(1) <= mutation_factor:
            child[i] = child[i] + np.random.normal(0,1)
    # print("new child",child)
    return child

def crossover(population):
    # print("inside")
    print(np.zeros((0,n_vars)).shape)
    # print("Pop shape", population.shape[0])
    new_population = np.zeros((0,n_vars))
    for p in range(0,population.shape[0],2):
        p1f1 = population_random_most_fit_reduction(population)
        p2f2 = population_random_most_fit_reduction(population)
        # p3f3 = population_random_most_fit_reduction(population)
        best_parent = 0 if p1f1[1] > p2f2[1] else 1
        
        # max 4 children
        n_children = np.random.randint(1,4, 1)[0]
        children = np.zeros((n_children, n_vars))
        for child in range(n_children):
            # randomness to each child
            randomness = np.random.uniform(0,1)

            # each child combination of their parents
            children[child] = p1f1[0]*randomness+p2f2[0]*(1-randomness)

            # mutate child
            children[child] = mutate(children[child])

            new_population = np.vstack((new_population, children[child]))

    return new_population

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

from sklearn import preprocessing

for i in range(generation_number, max_generations+1):
    print(f"gen number: {i}. population {population.shape}")
    children = crossover(population)
    fitness_children = evaluate(children)
    population = np.vstack((population, children))
    fitness_pop = np.append(fitness_pop, fitness_children)
    best_in_pop = np.argsort(fitness_pop)[-2:]

    # eliminate bad population
    # population = eliminate_bad_population(population)
    fit_pop_cp = fitness_pop
    print("fitness pop cp: ", fit_pop_cp.shape, fit_pop_cp)
    # min max scaling
    fit_pop_norm = preprocessing.minmax_scale(fitness_pop)  
    print("fitpopnorm: ", fit_pop_norm.shape, fit_pop_norm)
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    print("probs: ", probs)
    # 3 is n_population
    chosen = np.random.choice(population.shape[0], pop_size-2 , p=probs, replace=False)
    print("chosen: ", chosen)
    # manually add the best of the population
    chosen = np.append(chosen,best_in_pop[-1])
    chosen = np.append(chosen,best_in_pop[-2])
    print("chosen2: ", chosen)
    population = population[chosen]
    print(population.shape)
    fitness_pop = fitness_pop[chosen]

    # best, mean, std calculation
    best_in_pop = np.argsort(fitness_pop)[-2:]
    mean, std = np.mean(fitness_pop), np.std(fitness_pop)

    # saves the results for population
    text_file = open(experiment_name+"/pep_results.txt", "a")
    n = text_file.write(f'\n\nBest:{fitness_pop[best_in_pop[-1]]} \nSecond best:{fitness_pop[best_in_pop[-2]]}\nMean:{mean}\nstd:{std}\ngeneration: {i}\nBest_pop: {population[best_in_pop[-1]]} ')
    text_file.close()
    

    # saves simulation state
    solutions = [population, fitness_pop]
    env.update_solutions(solutions)
    env.save_state()


end_time = time.time() 
print(f"Time was {str(round((end_time-start_time)/60))} minutes")

env.state_to_log() 
print(f"Population shape: {population.shape}")
print(f"Fitness pop: {fitness_pop}")
print(f"Best: {best_in_pop[-1]}, Second best: {best_in_pop[-2]}")


