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

for hh in range(1,11):
    experiment_name = f'experiment_2parents_old_fitness_NEW_{hh}_en_1'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[1],
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
    ea_type = "2par_oldfit" # 3par_ownfit or 2par_oldfit

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 50
    mutation = 0.008
    last_best = 0

    # early stopping parameter after 15 rounds
    stopping = False
    early_stopping_rounds = 50

    use_new_f = False

    # runs simulation
    def simulation(env,x, test=False):
        f,p,e,t = env.play(pcont=x)
        
        # if use_new_f:
        #     f = 25**((1001 - t)/1000) + 75**((p-e)/100)
    #     if test == True:
    #         indiv_gain = p - e
    #         return (f2, indiv_gain)
    #     print("Own cool fitness score: {}".format(f2))
    #     return f2
        if test == True:
            indiv_gain = p - e
            return (f, indiv_gain)
        print("Old cool fitness score: {}".format(f))
        return f

    def evaluate(x, test=False):
        return np.array([simulation(env,y,test) for y in x])


    # tournament
    # randomly select two indivs out of pop and return the fittest
    def tournament(pop):
        
        # c1 = np.random.randint(0,pop.shape[0],1)
        # c2 = np.random.randint(0,pop.shape[0],1)

        # # get different parents
        # while c2 == c1:
        #     c2 = np.random.randint(0,pop.shape[0],1)
        # c3 = np.random.randint(0,pop.shape[0],1)
        # while c3 == c1 or c3 == c2:
        #     c3 = np.random.randint(0,pop.shape[0],1)

        # best = np.argmax([fit_pop[c1],fit_pop[c2],fit_pop[c3]])

        # if best == 0:
        #     return pop[c1][0], c1
        # elif best == 1:
        #     return pop[c2][0], c2
        # elif best == 2:
        #     return pop[c3][0], c3
        # else:
        #     print("Out of best array")
        #     return False

        
        c1 = np.random.randint(0,pop.shape[0],1)
        c2 = np.random.randint(0,pop.shape[0],1)

        # get different parents
        while c2 == c1:
            c2 = np.random.randint(0,pop.shape[0],1)
        best = np.argmax([fit_pop[c1],fit_pop[c2]])
        if best == 0:
            return pop[c1][0], c1
        elif best == 1:
            return pop[c2][0], c2
        else:
            print("Out of best array")
            return False
        # else:
        #     print("Wrong ea type specified, Choose 3par_ownfit or 2par_oldfit")



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

    
        # for p in range(0,pop.shape[0], 2):
        #     p1, pos_1 = tournament(pop)
        #     p2, pos_2 = tournament(pop)
        #     while pos_1 == pos_2:
        #         p2, pos_2 = tournament(pop)

        #     fit_1 = fit_pop[pos_1]
        #     fit_2 = fit_pop[pos_2]

        #     n_offspring =   np.random.randint(1,4, 1)[0]
        #     offspring =  np.zeros( (n_offspring, n_vars) )

        #     for f in range(0,n_offspring):
        #         randomness = np.random.dirichlet(np.ones(2),size=1)[0]
                
        #         offspring[f] = p1*float(randomness[0])+\
        #                     p2*float(randomness[1])

        #         # mutation
        #         for i in range(0,len(offspring[f])):
        #             if np.random.uniform(0 ,1)<=mutation:
        #                 chance = np.random.uniform(0, 1)
        #                 if chance <= 0.34:
        #                     offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)
        #             # mutate by swapping two genes (own addition)
        #                 elif 0.34 < chance <= 0.67:
        #                     rand_index = np.random.randint(0,len(offspring[f]))

        #                     temp_val = offspring[f][i]
        #                     offspring[f][i] =   offspring[f][rand_index]
        #                     offspring[f][rand_index] = temp_val
        #             # deletion of gene
        #                 else:
        #                     offspring[f][i] = 0

        #         offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

        #         total_offspring = np.vstack((total_offspring, offspring[f]))
        # return total_offspring


    # loads file with the best solution for testing
    if run_mode =='test':
        for i in range(5):
            bsol = np.loadtxt(experiment_name+'/best.txt')
            print( '\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('speed','normal')
            fpop_ig_array = evaluate([bsol], True)
            print(f"Fitness : {fpop_ig_array[0][0]}\nIndividual Gain: {fpop_ig_array[0][1]}")
            file_best  = open(experiment_name+'/best_ig.txt','a')
            file_best.write('instance individual_gain') if i == 0 else None
            file_best.write(f'\n {i} {fpop_ig_array[0][1]}')
            file_best.close()
        continue


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
        if i>4:
            use_new_f = True

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

    # The world in which we live today is created by evolution. Evolution in our world is a tedious process which takes many thousands of years, but, as a result it creates solutions to problems in ways that couldn't have been thought of before. **Good small example of evolution**. Machine learning is one of the prime examples where intelligent mathematics can construct a general problem solving machine. A problem with machine learning is that the solution to the problem will never be original. It usually uses the gradient to always move in the right direction. Although it provides robust problem solving routine, it is not similar to the most robust problem solving algorithm that we know: Evolution. Evolutionary Computing uses the basics of evolution (crossover, mutation, and recombination) to provide an original optimum solution to a given problem. Given the fact that real evolution took 1000's of years it can be imagined that evolutionary computing program work as fast as Moore's Law.