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
from sklearn import preprocessing

experiment_name = 'pep_spec'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode='ai',
                  player_controller=player_controller(n_hidden),
                  level=2)

# individual gain = player_energy - enemy_energy
# sol = np.loadtxt('solutions_demo/demo_all.txt')

n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

pop_size = 30
# num_pop=4
max_generations = 16
mutation_factor = 0.7
print(f"Vars are {n_vars}.")

start_time = time.time()

def simulation(env,x):
    f,*_ = env.play(pcont=x)
    # print(f"simulation {f},{p},{e},{t}")
    return f

def evaluate(x):
    return np.array([simulation(env,y) for y in x])

# randomly select two indivs out of pop and return the fittest
def tournament_selection(pop):
    c1 = np.random.randint(0,pop.shape[0],1)
    c2 = np.random.randint(0,pop.shape[0],1)

    # get different parents
    while c2 == c1:
        c2 = np.random.randint(0,pop.shape[0],1)
    c3 = np.random.randint(0,pop.shape[0],1)
    while c3 == c1 or c3 == c2:
        c3 = np.random.randint(0,pop.shape[0],1)

    best = np.argmax([fitness_pop[c1],fitness_pop[c2],fitness_pop[c3]])

    if best == 0:
        return (pop[c1][0],fitness_pop[c1])
    elif best == 1:
        return (pop[c2][0],fitness_pop[c2])
    elif best == 2:
        return (pop[c3][0],fitness_pop[c3])
    else:
        print("Out of best array")
        return False
    # return (pop[c1][0],fitness_pop[c1]) if fitness_pop[c1] > fitness_pop[c2] else (pop[c2][0],fitness_pop[c2])

# for every gen randomly choose to mutate or not
def mutate(child):
    for i in range(len(child)):
        if np.random.rand(1) <= mutation_factor:
            child[i] = child[i] + np.random.normal(0,1)
    return child

def crossover(population):
    print(np.zeros((0,n_vars)).shape)
    # print("Pop shape", population.shape[0])
    new_population = np.zeros((0,n_vars))

    for pop in range(1,int(population.shape[0]/2)):

        p1f1 = tournament_selection(population)
        p2f2 = tournament_selection(population)
        p3f3 = tournament_selection(population)

        best_parent = np.argmax([p1f1[1],p2f2[1],p3f3[1]])

        # max 4 children
        n_children = np.random.randint(1,4, 1)[0]
        children = np.zeros((n_children, n_vars))
        for child in range(n_children):
            # 3 random probs add up to 1
            randomness = np.random.dirichlet(np.ones(3),size=1)[0]

            # each child combination of their parents
            children[child] = p1f1[0]*float(randomness[0])+\
                              p2f2[0]*float(randomness[1])+\
                              p3f3[0]*float(randomness[2])

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

for i in range(generation_number, max_generations+1):
    print(f"gen number: {i}. population {population.shape}")
    children = crossover(population)
    fitness_children = evaluate(children)
    population = np.vstack((population, children))
    fitness_pop = np.append(fitness_pop, fitness_children)
    best_in_pop = np.argsort(fitness_pop)[-2:]

    # eliminate bad population
    fit_pop_cp = fitness_pop

    # min max scaling
    fit_pop_norm = preprocessing.minmax_scale(fitness_pop)  
    probs = (fit_pop_norm)/(fit_pop_norm).sum()

    # choose new population based on fitness
    chosen = np.random.choice(population.shape[0], pop_size-2 , p=probs, replace=False)

    # elitism
    chosen = np.append(chosen,best_in_pop[-1])
    chosen = np.append(chosen,best_in_pop[-2])
    
    population = population[chosen]
    fitness_pop = fitness_pop[chosen]

    # decrease mutation factor every 5 generations until 0.2
    if i%5 == 0 and mutation_factor > 0.2:
        mutation_factor -= 0.1

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


