'''implements a genetic algorithm to find a TSP solution:
gene: city (x,y)
population: set of routes from start to end of given population size
parents: two routes combined to create new route
mating pool: collection of parents used to create new routes
fitness: evaluation measure for route
mutation: random city-swap in route
elitism: mechanism for carrying best indiviudals into next generation

enables use of one of the two proposed cross-over functions (ordered crossover, CX1, and CX2)
from Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator
in https://www.hindawi.com/journals/cin/2017/7430125/)
'''
from __future__ import annotations
import sys
sys.path.append('./utils')
import copy
import aco
import random
import numpy as np
import operator
from tqdm import tqdm


class Fitness(object):
    def __init__(self, cost_matrix:np.ndarray):
        self.cost_matrix  = cost_matrix

    def calc_route_length(self, route:list):
        # use array calc
        #print(route)
        end_end_distance = np.sum(self.cost_matrix[route[:-1], route[1:]])
        # add travel cost from end-city to start-city
        end_end_distance += self.cost_matrix[route[-1], route[0]]
        return float(end_end_distance)


    def calc_route_fitness(self, route:list):
        return 1/ self.calc_route_length(route)

class Genetic(object):

    def __init__(self, graph:aco.Graph, mutation_rate:float, population_size:int,
                 num_gen:int, elite_size:float, crossover='ordered'):
        random.seed(1234567)
        self.cost_matrix = copy.deepcopy(graph.matrix)
        self.rank = self.cost_matrix.shape[0]
        np.fill_diagonal(self.cost_matrix, float('inf'))
        self.fitness = Fitness(cost_matrix=self.cost_matrix)
        self.mutation_rate = mutation_rate
        self.elite_size = int(elite_size*population_size)
        self.crossover_type = crossover
        self.generations = num_gen
        self.population_size = population_size
        self.change = []

    def _create_route(self):
        route = random.sample(list(range(self.rank)), self.rank)
        return route

    def _init_population(self):
        popul = []
        for i in range(self.population_size):
            popul.append(self._create_route())

        return popul

    def _calc_route_ranking(self, popul:list):
        fitness = {}
        for r_id in range(len(popul)):
            fitness[r_id] = self.fitness.calc_route_fitness(popul[r_id])

        return(sorted(fitness.items(), key=operator.itemgetter(1), reverse=True))

    def _tournament_selection(self, sorted_fitness:list, k:int):
        # implements tournament selction for parents
        shuffled = random.sample(sorted_fitness, k=len(sorted_fitness))
        if k>int(len(sorted_fitness)/2):
            k = int(len(sorted_fitness)/2)
        n = int(len(sorted_fitness)/k)
        parents = [(-1,float('inf')) for _ in range(k)]
        plc = 0
        for i, route in enumerate(shuffled):
            if i>n-1:
                plc += 1
            if route[1] < parents[plc][1]:
                parents[plc] = route

        return parents

    def _fitness_proportional_selection(self, sorted_fitness:list):
        # implements proportionate selection via fitness-weighted probability
        selection = []
        #elite selection
        for i in range(self.elite_size):
            selection.append(sorted_fitness[i][0])

        #roulette wheel
        srt = np.array(sorted_fitness, dtype=float)
        srt[:,1] = srt[:,1] / np.sum(srt[:,1])
        inds = np.random.choice(srt[:,0], size=(len(sorted_fitness)-self.elite_size), replace=True, p = srt[:,1])
        inds = inds.astype(int)
        selection.extend(inds.tolist())
        return selection

    def _mating_pool(self, popul:list, selection:list):
        pool = [popul[r_ind] for r_ind in selection]
        return pool

    def _cx1_helper(self, par1:list, par2:list, result:np.ndarray, val_ind:int, cycl:set):
        # recursive function to make the first crossover cycle
        if val_ind < len(par2):
            val = par2[val_ind]
            if val not in cycl:
                plc = par1.index(val)
                result[plc] = val
                cycl.add(val)
                return self._cx1_helper(par1=par1, par2=par2, result=result, val_ind=plc, cycl=cycl)
            else:
                # cycle is completed
                return result
        else:
            return result

    def _cx2_helper(self, par1:list, par2:list, offspring1:np.ndarray,
                    offspring2:np.ndarray, val_ind1:int, cycl1:set, cycl2:set, next_ind:int):

        if par2[val_ind1] not in cycl1 and next_ind < len(par1):
            offspring1[next_ind] = par2[val_ind1]
            cycl1.add(par2[val_ind1])
            plc2, val_ind1 =self._two_step_index(par2[val_ind1], par1, par2)
            if plc2 not in cycl2:
                offspring2[next_ind] = plc2
                cycl2.add(plc2)
                return self._cx2_helper(par1, par2, offspring1, offspring2, val_ind1, cycl1, cycl2, next_ind+1)
            elif next_ind < len(par1):
                adds1, adds2 = self._cx2_fill(par1, par2, offspring1, offspring2)
                offspring1[next_ind:len(par1)] = adds1
                offspring2[next_ind:len(par1)] = adds2

                return offspring1, offspring2
            else:
                return offspring1, offspring2

        elif next_ind < len(par1):
            adds1, adds2 = self._cx2_fill(par1, par2, offspring1, offspring2)
            offspring1[next_ind:len(par1)] = adds1
            offspring2[next_ind:len(par1)] = adds2

            return offspring1, offspring2
        else:
            return offspring1, offspring2

    def _cx2_fill(self, par1:list, par2:list, offspring1:np.ndarray, offspring2:np.ndarray):
        #fill remnaining when cycle comes to end prematurely
        ofset1 = set(offspring1)
        ofset2 = set(offspring2)
        par1 = [el for el in par1 if el not in ofset2]
        par2 = [el for el in par2 if el not in ofset1]
        adds1 = [par2[0]]
        adds2 = []
        val, val_ind1 = self._two_step_index(adds1[0], parent1=par1, parent2=par2)
        adds2.append(val)
        for _ in range(1, len(par1)):
            #complete cycle
            adds1.append(par2[val_ind1])
            val, val_ind1 = self._two_step_index(par2[val_ind1], parent1=par1, parent2=par2)
            adds2.append(val)

        return adds1, adds2

    def _two_step_index(self, val, parent1:list, parent2:list):
        val_ind = parent1.index(parent2[parent1.index(val)])
        return parent2[val_ind], parent1.index(parent2[val_ind])

    def _crossover(self, parent1:list, parent2:list, typ='cx1'):
        parents = [parent1, parent2]
        start = random.randint(0,1)
        end = 1 - start
        parents = [parents[start], parents[end]]
        result = [np.zeros(shape=(len(parent1),)) for _ in range(len(parents))]
        for el in result:
            el.fill(-1)

        if typ == 'cx1':
            result = self._crossover_cx1(parents, result)
        elif typ == 'cx2':
            result = self._crossover_cx2(parents, result)
        else:
            result = self._ordered_crossover(parents, result)
        return [list(el) for el in result]

    def _crossover_cx1(self, parents:list, result:list):
        from collections import Counter
        #implements cx1 (a.k.a cycle crossover by Oliver et al.)
        result[0][0] = parents[0][0]
        #set to track cycle
        cycl = set(result[0])
        result[0] = self._cx1_helper(par1=parents[0], par2=parents[1],
                                        result=result[0], val_ind=0, cycl=cycl)
        result[0] = [int(result[0][i]) if result[0][i] != -1 else int(parents[1][i]) for i in range(len(parents[1]))]
        assert len(np.unique(result[0])) == len(parents[1])
        #repeat for second offspring
        result[1][0] = parents[1][0]
        # set to track cycle
        cycl = set(result[1])
        result[1] = self._cx1_helper(par1=parents[1], par2=parents[0],
                                     result=result[1], val_ind=0, cycl=cycl)
        result[1] = [int(result[1][i]) if result[1][i] != -1 else int(parents[0][i]) for i in range(len(parents[0]))]

        return result

    def _crossover_cx2(self, parents:list, result:list):
        # implements the cx2 crossover suggested by Hussain et al. 2017
        result[0][0] = parents[1][0]
        result[1][0], val_ind1 = self._two_step_index(result[0][0], parent1=parents[0], parent2=parents[1])
        # set to track cycle
        cycl1 = set(result[0][0])
        cycl2 = set(result[1][0])

        res1, res2 = self._cx2_helper(par1=parents[0], par2=parents[1],
                                   offspring1=result[0], offspring2=result[1],
                                   val_ind=val_ind1, cycl1=cycl1, cycl2=cycl2, next_ind=0+1)
        result[0] = res1
        result[1] = res2
        return result

    def _ordered_crossover(self, parents:list, result:list):
        #implements Davis ordered crossover
        offsprings = result
        # define two random cutpoints
        cut1 = random.randint(1, len(parents[0]))
        cut2 = random.randint(1, len(parents[0]))
        cut1, cut2 = sorted([cut1, cut2])

        offsprings[0][cut1:cut2] = parents[0][cut1:cut2]
        offsprings[1][cut1:cut2] = parents[1][cut1:cut2]

        cycl1 = set(offsprings[0].tolist())
        cycl2 = set(offsprings[1].tolist())
        remaind1 = [el for el in parents[1][cut2:] + parents[1][:cut2] if el not in cycl1]
        remaind2 = [el for el in parents[0][cut2:] + parents[0][:cut2] if el not in cycl2]
        offsprings[0][cut2:] = remaind1[:(len(parents[0])-cut2)]
        offsprings[1][cut2:] = remaind2[:(len(parents[0]) - cut2)]  # fill end of child
        offsprings[0][:cut1] = remaind1[(len(parents[0]) - cut2):]  # fill start of child
        offsprings[1][:cut1] = remaind2[(len(parents[0]) - cut2):] # fill start of child
        offsprings = [el.astype(int) for el in offsprings]
        return offsprings
        
    def _generate_population_offspring(self, pool:list, crossovertype='ordered'):
        #generates next generation from parent population

        #select elite definetly
        offspring = copy.deepcopy(pool[:self.elite_size])
        #select random sample for mating
        randsize = len(pool)-self.elite_size
        #make sure even
        randsize = randsize if randsize%2 == 0 else randsize+1
        sample = random.sample(pool, k=randsize)
        for i in range(0, len(sample), 2):
            par1 = sample[i]
            par2 = sample[i+1]
            children = self._crossover(parent1=par1, parent2=par2, typ=crossovertype)
            offspring.extend(children)

        assert len(offspring) >= len(pool) # make sure we don't become extinct here
        return offspring

    def _swap_mutate(self, individual):
        for do_swap in range(len(individual)):
            if(random.random() < self.mutation_rate):
                exchange = random.randint(0, len(individual)-1)
                tmp = individual[exchange]
                individual[exchange] = individual[do_swap]
                individual[do_swap] = tmp

        return individual

    def _mutate_population(self, popul:list):
        mutated = []
        for indiv in popul:
            mutated.append(self._swap_mutate(indiv))
        return mutated

    def _select_crossover_type(self):
        if self.crossover_type == 'random':
            return random.choice(['ordered', 'cx1', 'cx2'])
        else:
            return self.crossover_type
    
    def _produce_next_gen(self, current):

        ranked = self._calc_route_ranking(current)
        self.change.append(self.fitness.calc_route_length(current[ranked[0][0]]))
        selected = self._fitness_proportional_selection(ranked)
        matingPool = self._mating_pool(current, selected)
        offspring = self._generate_population_offspring(matingPool, crossovertype=self._select_crossover_type())
        nextGen = self._mutate_population(popul=offspring)
        return nextGen

    def genetic_solve(self):
        popul = self._init_population()

        for i in tqdm(range(self.generations)):
            popul = self._produce_next_gen(current=popul)


        ranked = self._calc_route_ranking(popul)
        dist = self.fitness.calc_route_length(popul[ranked[0][0]])
        self.change.append(dist)
        self.best_dist = dist
        self.best_result = ranked[0]
        self.population = popul
        return dist, popul[ranked[0][0]]
        


if __name__ == '__main__':
    random.seed(12345)
    cost = np.zeros(shape=(10, 10))
    tril = np.tril_indices_from(cost, k=-1)
    cost[tril] = np.arange(len(tril) + 1)
    cost.T[tril] = cost[tril]
    graph = aco.Graph(cost_matrix=cost)
    gen = Genetic(graph = graph, mutation_rate=0.02, population_size=100,
                  num_gen=100, elite_size=0.2, crossover='ordered')
    res = gen.genetic_solve()
    print('Best cost is {}'.format(res[1]))
        







