###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
from scipy.spatial import distance_matrix
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from sklearn import preprocessing

# turn off video
os.environ["SDL_VIDEODRIVER"] = "dummy"

for hh in range(1,2):
    experiment_name = f'Ass2_TEST13_2_elitism{hh}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for multiple enemies
    env = Environment(experiment_name=experiment_name,
                  enemies=[1,5,6],
                  multiplemode="yes",
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

    run_mode = 'train' # train or test or testall
    ea_type = "2par_oldfit" # 3par_ownfit or 2par_oldfit

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 50
    mutation = 0.02
    last_best = 0

    # fitness sharing sigma = how much space and alpha = multiplication factor
    fs_sigma = 10.0
    fs_alpha = 1.0

    # Blend crossover alpha
    alpha = 0.5

    # early stopping parameter after 15 rounds
    stopping = False
    early_stopping_rounds = 50
    use_new_f = False
    live_per = 0.0

    # old fitness:
    # return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - numpy.log(self.get_time())
    # new: return values.mean() - values.std()

    # runs simulation
    def simulation(env,x, test=False):
        f,p,e,t = env.play(pcont=x)
        # experiment with decreasing amounts
        # print(f,p,e,t,live_per)
        # if live_per <= 0.9:
        #     print("IN")
        #     if t >=0:
        #         f = (1.0-live_per)*(100 - e) + (live_per)*p - np.log(t)
        #     else:
        #         f = (1.0-live_per)*(100 - e) + (live_per)*p - t
        #     print((1.0-live_per),(100 - e),(live_per)*p, np.log(t))
        # print(f)
        if test == True:
            indiv_gain = p - e
            return (f, indiv_gain)
        # print("Old cool fitness score: {}".format(f))
        print(f"fitpop:{f}")

        # print(f.mean()-f.std(), f.mean(), f.std())
        return f

    def evaluate(x, test=False):
        return np.array([simulation(env,y,test) for y in x])
    
    def sharing(distance, sigma, alpha):
        res = 0
        times = 0
        if distance<sigma:
            if times>0:
                print("applied distance")
            res += 1 - (distance/sigma)**alpha
            times+=1
        return res

    def shared_fitness(fitness_pop, e_individual, population, sigma, alpha):
        num = fitness_pop[e_individual[0]]

        dists = distance_matrix([e_individual[1]], population)[0]
        tmp = [sharing(d, sigma, alpha) for d in dists]
        return num/sum(tmp),


    # tournament
    # randomly select two indivs out of pop and return the fittest
    def tournament(pop):       
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

            fit_1 = fit_pop[pos_1]
            fit_2 = fit_pop[pos_2]

            n_offspring =   np.random.randint(1,4, 1)[0]
            offspring =  np.zeros( (n_offspring, n_vars) )

            for f in range(0,n_offspring):
                # randomness = np.random.dirichlet(np.ones(2),size=1)[0]
                # offspring
                # offchance = np.random.uniform(0,1)
                # if offchance <= 0.34:
                #     offspring[f] = p1*float(randomness[0])+\
                #                    p2*float(randomness[1])
                # else:

                # blend crossover
                for i, (x1,x2) in enumerate(zip(p1,p2)):
                    min_par = np.minimum(p1[i],p2[i])
                    max_par = np.maximum(p1[i],p2[i])
                    maxmin = max_par-min_par
                    offspring[f][i] = np.random.uniform(min_par-alpha*maxmin,max_par+alpha*maxmin)


                # mutation
                for i in range(0,len(offspring[f])):
                    if np.random.uniform(0 ,1)<=mutation:
                        chance = np.random.uniform(0, 1)
                        if chance <= 0.34:
                            offspring[f][i] = offspring[f][i]+np.random.normal(0, 1)
                    # mutate by swapping two genes
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
    
    # change the genes of the 1/3 worst population randomly with genes of the 1/3 best
    def kill_worst_genomes(pop, fit_pop):
        arg_ordered_pop = np.argsort(fit_pop)
        worst_indivs = arg_ordered_pop[:int(npop/3)]
        for i in worst_indivs:
            for j in range(n_vars):
                n = np.random.randint(0,int(npop/3))
                pop[i][j] = pop[arg_ordered_pop[-n-1:]][0][j] 
        return pop, fit_pop

    # loads file with the best solution for testing
    if run_mode == 'test':
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
    elif run_mode == 'testall':
        for i in range(5):
            bsol = np.loadtxt(experiment_name+'/best.txt')
            print( '\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('speed','normal')
            env.update_parameter('enemies',[1,2,3,4,5,6,7,8])
            fpop_ig_array = evaluate([bsol], True)
            print(f"Fitness : {fpop_ig_array[0][0]}\nIndividual Gain: {fpop_ig_array[0][1]}")
            file_best  = open(experiment_name+'/best_ig_all.txt','a')
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

    final_best_indiv = [0, 0]
    old_best,old_sbest,old_tbest = 0,0,0
    for i in range(ini_g+1, gens):
        # if i>4:
        #     use_new_f = True
        if i % 10 == 0:
            print("add1")
            if live_per < 0.9:
                print("add2")
                live_per += 0.1
        
        # save best solution and repeat
        old_best = pop[np.argmax(fit_pop)]
        best_three = fit_pop.argsort()[-3:][::-1]
        old_sbest = pop[best_three[1]]  
        old_tbest = pop[best_three[2]]
        # print(old_best,old_sbest,old_tbest)
        # print(type(old_best),type(old_sbest),type(old_tbest))
        # print(type(pop), type(pop[0]))


        offspring = crossover(pop)  # crossover
        fit_offspring = evaluate(offspring)  # evaluation
        pop = np.vstack((pop,offspring))
        fit_pop = np.append(fit_pop,fit_offspring)


        # selection
        # fitness sharing
        fit_pop_fs = [shared_fitness(fit_pop, (e, i), pop, sigma=fs_sigma, alpha=fs_alpha) for e, i in enumerate(pop)]
        fit_pop_cp = np.array(fit_pop_fs)
        fit_pop = np.squeeze(fit_pop_cp)
        print(fit_pop, fit_pop.shape)

        best = np.argmax(fit_pop)
        fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])
        best_sol = fit_pop[best]

        # use sklearn minmax scaler
        fit_pop_norm = preprocessing.minmax_scale(fit_pop) 

        # percentagely choose the new population with elitism of the top 3 
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        print(pop.shape[0],npop)
        chosen = np.random.choice(pop.shape[0], npop-3 , p=probs, replace=False)
        # print(chosen.shape)
        pop = pop[chosen]
        pop = np.vstack((pop[1:],old_best))
        pop = np.vstack((pop[1:],old_sbest))
        pop = np.vstack((pop[1:],old_tbest))
        print("chosen",chosen)
        # pop = np.array(pop)

        fit_pop = fit_pop[chosen]


        # early stopping
        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= early_stopping_rounds:
            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\nreplacing_bad_individuals')
            file_aux.close()
            pop, fit_pop = kill_worst_genomes(pop, fit_pop)
            notimproved = 0

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

        # saves file with the best solution if its best
        if fit_pop[best] > final_best_indiv[0]:
            np.savetxt(experiment_name+'/best.txt',pop[best])
            file_aux  = open(experiment_name+'/best_fit.txt','a')
            file_aux.write(f'Best fitness is now {fit_pop[best]}\n{pop[best]}\n\n')
            file_aux.close()
            final_best_indiv[0] = fit_pop[best]
            final_best_indiv[1] = pop[best]

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