###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from sklearn import preprocessing


experiment_name = 'extended crossover lower mutate rate random doomsday own fitness  v2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 50
gens = 100
mutation = 0.008
last_best = 0

# early stopping parameter after 15 rounds
stopping = False
early_stopping_rounds = 15


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)

    f = (1001 - t)/1000 *(p-e)
    print("Own cool fitness score: {}".format(f))
    return f

# normalizes
# def norm(x, pfit_pop):

#     if ( max(pfit_pop) - min(pfit_pop) ) > 0:
#         x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
#     else:
#         x_norm = 0

#     if x_norm <= 0:
#         x_norm = 0.0000000001
#     return x_norm


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# tournament
# def tournament(pop):
#     fittest =  np.random.randint(0,pop.shape[0], 1)

#     for _ in range(np.random.randint(1,pop.shape[0])):
#         c2 =  np.random.randint(0,pop.shape[0], 1)
#         if fit_pop[c2] > fit_pop[fittest]:
#             fittest = c2

#     return pop[fittest][0], fittest

# randomly select two indivs out of pop and return the fittest
def tournament(pop):
    c1 = np.random.randint(0,pop.shape[0],1)
    c2 = np.random.randint(0,pop.shape[0],1)

    # get different parents
    while c2 == c1:
        c2 = np.random.randint(0,pop.shape[0],1)
    c3 = np.random.randint(0,pop.shape[0],1)
    while c3 == c1 or c3 == c2:
        c3 = np.random.randint(0,pop.shape[0],1)

    best = np.argmax([fit_pop[c1],fit_pop[c2],fit_pop[c3]])

    if best == 0:
        return pop[c1][0], c1
    elif best == 1:
        return pop[c2][0], c2
    elif best == 2:
        return pop[c3][0], c3
    else:
        print("Out of best array")
        return False


# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


# crossover
def crossover(pop):

    total_offspring = np.zeros((0,n_vars))


    for p in range(0,pop.shape[0], 2):
        p1, pos_1 = tournament(pop)
        p2, pos_2 = tournament(pop)
        while pos_1 == pos_2:
            p2, pos_2 = tournament(pop)
        p3, pos_3 = tournament(pop)
        while pos_2 == pos_3:
            p3, pos_3 = tournament(pop)


        fit_1 = fit_pop[pos_1]
        fit_2 = fit_pop[pos_2]
        fit_3 = fit_pop[pos_3]

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars) )

        for f in range(0,n_offspring):
            randomness = np.random.dirichlet(np.ones(3),size=1)[0]
            
            offspring[f] = p1*float(randomness[0])+\
                           p2*float(randomness[1])+\
                           p3*float(randomness[2])

            # mutation
            for i in range(0,len(offspring[f])):
                if np.random.uniform(0 ,1)<=mutation:
                    chance = np.random.uniform(0, 1)
                    if chance <= 0.34:
                        offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)
                # mutate by swapping two genes (own addition)
                    elif 0.34 < chance <= 0.67:
                        rand_index = np.random.randint(0,len(offspring[f]))

                        temp_val = offspring[f][i]
                        offspring[f][i] =   offspring[f][rand_index]
                        offspring[f][rand_index] = temp_val
                # deletion of gene
                    else:
                        offspring[f][i] = 0

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


# kills the worst genomes, and replace with new best/random solutions
# def doomsday(pop,fit_pop):

#     worst = int(npop/4)  # a quarter of the population
#     order = np.argsort(fit_pop)
#     orderasc = order[0:worst]

#     for o in orderasc:
#         for j in range(0,n_vars):
#             pro = np.random.uniform(0,1)
#             if np.random.uniform(0,1)  <= pro:
#                 pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
#             else:
#                 pop[o][j] = pop[order[-1:]][0][j] # dna from best

#         fit_pop[o]=evaluate([pop[o]])

#     return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    offspring = crossover(pop)  # crossover
    fit_offspring = evaluate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop

    # use sklearn minmax scaler
    fit_pop_norm = preprocessing.minmax_scale(fit_pop)  
    # fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= early_stopping_rounds:
        print("Early stopping activated")
        stopping = True
        # file_aux  = open(experiment_name+'/results.txt','a')
        # file_aux.write('\ndoomsday')
        # file_aux.close()

        # pop, fit_pop = doomsday(pop,fit_pop)
        # notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

    if stopping == True:
        break




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
