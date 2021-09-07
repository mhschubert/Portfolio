#this aco includes the optimizations proposed by Zhu et al (2020)
#published in artificial intelligence 2020 https://doi.org/10.1007/s10489-020-01841-x <- lots of changes when compared to the version on ieee
from __future__ import annotations
import sys
import random
from collections import Counter
import numpy as np
from scipy import stats
import math
import copy
sys.path.append('./utils')
from utils import helper, prim_min, prim_max
from tqdm import tqdm
import multiprocessing as mp

class Graph(object):
    def __init__(self, cost_matrix: np.ndarray):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        np.fill_diagonal(cost_matrix, float('inf'))
        self.matrix = cost_matrix
        self.eta = 1/self.matrix
        np.fill_diagonal(self.eta, 0)
        np.fill_diagonal(self.matrix, 0)
        self.rank = self.matrix.shape[0]

        self.initial_pheromone = 1 / (self.rank * self.rank)
        self.pheromone = np.zeros(shape=(self.rank, self.rank))
        self.pheromone.fill(self.initial_pheromone)
        np.fill_diagonal(self.pheromone, 0)



class ACO(object):

    def __init__(self, num_ants:int, generations:int, alpha:float, beta:float,
                rho:float, psi:float, rho_span_min:float, rho_span_max:float, update_rule:float, q0 = 0.5, min_inds=None,
                max_inds = None, multiprocess = True):
        '''Here we specify the values for the algorihm
        alpha = weighting parameter for pheromones
        beta = weighitng parameter for distance-cost
        rho = global pheromone decay/evaporation parameter
        psi = local pheromone decay/evaporation parameter
        rho_span_min = pheromone weighting paramter for the minimal span tree path
        rho_span_max = pheromone weighting paramter for the maximum span tree path
        h0 = threshold for min-span tree similarity
        q0 = threshold for pribabilisitc ant in acs and dgm
        update_rule: 1 = acs, 2=mmas, 3=dgmaco
        min_inds/max_inds = respective span tree indices -> not necessary, is calculated when using ACO as stand-alone and calling on solve()
        '''
        random.seed(12345)
        self.num_ants = num_ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.psi = psi
        self.rho_span_min = rho_span_min
        self.rho_span_max = rho_span_max
        self.rule = update_rule
        self.min_st_inds = min_inds
        self.max_st_inds = max_inds
        self.q0 =q0
        self.mulitprocess = multiprocess

    def _setup_solver_variables(self):
        self.best_cost = float('inf')
        self.best_ants = []
        self.best_solution = []
        self.best_ants_delta = []
        self.avg_costs = []
        self.best_costs = []
        self.cost_tracker = []
        self.round_tracker = -1

    def solve(self, graph:Graph):
        self.best_cost = float('inf')
        self.best_ants = []
        self.best_solution = []
        self.cost_tracker = []
        self.round_tracker = -1
        self.best_ants_delta = []
        self.avg_costs = []
        self.best_costs = []
        self.min_st_inds = [[],[]]
        self.max_st_inds = [[], []]
        self.min_st_inds[0], self.min_st_inds[1] = prim_min.get_minimum_cost(graph.matrix)
        self.max_st_inds[0], self.max_st_inds[1] = prim_max.get_maximum_cost(graph.matrix)
        for gen in tqdm(range(self.generations)):
            self._sovlve_gen(graph, gen)
        return self.best_solution, self.best_cost, self.avg_costs, self.best_costs
    def _calc_ant_multi(self, ant:_Ant):
        ant.select_route()
        return ant

    def _sovlve_gen(self, graph:Graph, gen:int):
        graph.prev_pheromones = graph.pheromone.copy()
        ants = [_Ant(aco=self ,graph=graph) for i in range(self.num_ants)]
        if self.rule == 3:
            self._dgmaco_update(graph, self.min_st_inds, self.max_st_inds, gen, self.generations)
        curr_costs = []
        if self.mulitprocess != True:
            for i, ant in enumerate(ants):
                ant.select_route()
        else:
            with mp.Pool(mp.cpu_count()-1) as pool:
                ants = pool.map(self._calc_ant_multi, ants)
        for i, ant in enumerate(ants):
            curr_costs.append(ant.total_cost)
            if ant.total_cost <= self.best_cost:
                #keep only best solutions from one generation if better than previous
                if gen != self.round_tracker:
                    self.best_solution = []
                    self.best_ants = []
                    self.best_ants_delta = []
                    self.round_tracker = gen
                self.best_cost = ant.total_cost
                self.best_solution.append(ant.forbidden)
                self.best_ants.append(ant)
                self.best_ants_delta.append(ant.pheromone_delta)

        self.ants = ants
        self._update_pheromones(graph, self.best_ants_delta, ants, self.best_cost, gen)
        self.avg_costs.append(np.mean(curr_costs))
        self.cost_tracker.append(curr_costs)
        self.best_costs.append(self.best_cost)


    def _update_pheromones(self, graph:Graph, best_ants_delta:list, ants:list, best_cost:float,n:int):
        # pheromone updates
        self._update_pheromone_global(graph, best_ants_delta, best_cost, n)
        if self.rule != 2:
            # local updates if not min-max
            for ant in ants:
                graph.pheromone = graph.pheromone + ant.pheromone_delta

    def _update_pheromone_global(self, graph:Graph, best_ants_delta:list, best_cost:float,n:int):
        if self.rule == 1 or self.rule == 3:
            self._acs_global_update(graph, best_ants_delta, best_cost)
        elif self.rule == 2:
            self._min_max_update(graph, best_ants_delta, best_cost, n)

    def _acs_global_update(self, graph:Graph, best_ants_delta, best_cost):
        #get the inds for all pheromone_deltas
        inds = helper.get_all_nonzeros(best_ants_delta)
        # artific_intell paper eq.2
        graph.pheromone = (1-self.rho)*graph.prev_pheromones
        graph.pheromone[inds[0], inds[1]] += self.rho * (1/best_cost)
        graph.pheromone[inds[1], inds[0]] += self.rho * (1 / best_cost)

    def _min_max_update(self, graph:Graph, best_ants_delta, best_cost, n):
        #get the inds for all pheromone_deltas
        #set round counter to natural counting
        n += 1
        nzer = np.zeros(shape=best_ants_delta[0].shape)
        for el in best_ants_delta:
            nzer += el
        indsz = nzer.nonzero()
        #artific_intell paper eq.5-7
        graph.pheromone = self.rho*graph.pheromone
        #1/best_cost is too small - hence we use 1/best_cost * 10 ** 5 <- hack because we operate with very large costs/distances
        graph.pheromone[list(indsz[0]), list(indsz[1])] += ((1/best_cost) * 10 **5)
        graph.pheromone[list(indsz[1]), list(indsz[0])] += ((1/best_cost) * 10 **5)

        tau_max = (1/(1-self.rho)) * ((1/best_cost) * 10 **5)
        denom = (((n/2) - 2) * (0.05 ** (1/n)))
        if denom >=1:
            tau_min = (tau_max * (1.0 - 0.05 ** (1.0/n))) / denom
        else:
            tau_min = tau_max / (10**6)

        graph.pheromone[graph.pheromone < tau_min] = tau_min
        graph.pheromone[graph.pheromone > tau_max] = tau_max

    def _dgmaco_update(self, graph:Graph, min_st, max_st, n, N):
        #implements the spantree update in artific_intell paper eqs.11
        if n > N/3:
            return
        graph.pheromone[min_st[0], min_st[1]] = self.rho_span_min * graph.initial_pheromone * (n/N)
        graph.pheromone[min_st[1], min_st[0]] = self.rho_span_min * graph.initial_pheromone * (n/N)

        #implements eq. 12
        graph.pheromone[max_st[0], max_st[1]] = self.rho_span_max * graph.initial_pheromone * ((N-n)/N)
        graph.pheromone[max_st[1], max_st[0]] = self.rho_span_max * graph.initial_pheromone * ((N - n) / N)

class _Ant(object):

    def __init__(self, aco:ACO, graph:Graph):
        self.graph = graph
        self.aco = aco
        self.alpha = np.full(shape=self.graph.matrix.shape, fill_value= self.aco.alpha)
        self.beta = np.full(shape=self.graph.matrix.shape, fill_value=self.aco.beta)
        self.eta_beta = self.graph.eta ** self.beta
        np.fill_diagonal(self.eta_beta, 0)
        self.total_cost = 0.0
        self.forbidden = []
        self.pheromone_delta = np.zeros(shape=self.graph.matrix.shape)
        self.allowed = [i for i in range(graph.rank)]
        start = random.randint(0, graph.rank-1)
        self.current = start
        self.forbidden.append(self.current)
        self.allowed.remove(self.current)

    def select_route(self):
        #calculate weight matrices first
        # create probability matrix
        p_t = np.zeros(shape=(self.graph.matrix.shape))
        #weight matrix for probabilistic next step (eq. 3)

        weight_matrix = np.multiply(
            self.graph.pheromone ** self.aco.alpha,
            self.eta_beta)
        np.fill_diagonal(weight_matrix, 1)
        #next step matrix for directed next step (eq. 2 and eq. 11)
        if self.aco.rule == 1 or self.aco.rule == 2:
            argmax_pick = self._traditional_best_pick()
        else:
            argmax_pick = self._udpo_best_pick()
        np.fill_diagonal(argmax_pick, 0)
        #generate array with random q
        q = np.random.uniform(0.0, 1.0, size=self.graph.rank-1) #exclude start
        #iterate over the cities we have to visit
        for stop in q:
            #eq. 2 and eq.3 test
            if stop <=self.aco.q0 and len(self.allowed)>1:
                mx = np.max(argmax_pick[self.current, self.allowed])
                inds = np.where(argmax_pick[self.current, :] == mx)[0].tolist()
                inds = [ind for ind in inds if ind in set(self.allowed)]
                ind = inds[0]
            elif len(self.allowed)>1:
                #we iterate row-wise over p_t. Thus we do not need to reset the matrix
                weight_matrix[np.isnan(weight_matrix)] = 0
                weight_matrix[np.isinf(weight_matrix)] = np.max(weight_matrix[~np.isinf(weight_matrix)])
                weight_sum = np.sum(weight_matrix[self.current, self.allowed])
                if weight_sum == 0:
                    weight_sum += 1*(10**-6)

                p_t[self.current, self.allowed] = weight_matrix[self.current, self.allowed]/ weight_sum
                #if weight is mapped to zero cause weight too small
                p_t[self.current, self.allowed] = np.nan_to_num(p_t[self.current, self.allowed], nan=0.999999)
                #make probability array for current iteration
                tmp = p_t[self.current, :].flatten()
                if np.sum(tmp < 0) >0:
                    nzer = np.nonzero(tmp)
                    tmp[nzer] -= np.min(tmp)
                tmp = tmp/np.sum(tmp)


                #select ind of next step
                try:
                    ind = np.random.choice(range(self.graph.rank), replace=False, size=1, p=tmp)[0]
                except Exception as e:
                    #some error management
                    print('colony {}'.format(self.aco.rule))
                    print(stop)
                    if stop <=self.aco.q0:
                        print(argmax_pick)
                    else:
                        print(weight_matrix)
                        print(weight_sum)
                    print('tmp')
                    print(tmp)
                    print(self.graph.pheromone)
                    print('vals in weight matrix allowed')
                    print(weight_matrix[self.current, self.allowed])
                    print(self.allowed)
                    raise e
                    sys.exit()
            else:
                assert len(self.allowed) == 1
                ind = self.allowed[0]
            #update the current state and the allowed
            try:
                self.allowed.remove(ind)
                self.forbidden.append(ind)
                self.total_cost += self.graph.matrix[self.current, ind]
            except Exception as e:
                #some error management
                print('colony {}'.format(self.aco.rule))
                print(ind)
                print(self.allowed)
                print(self.forbidden)
                raise e
                sys.exit()
            #do it here to safe computation time lateron
            self._pheromone_update(self.current, ind)
            self.current = ind
        # we do a circular tour - so we have to add the direction last-first to the solution
        self._make_circular()

    def _make_circular(self):
        # makes the one-way tour circular
        start = self.forbidden[0]
        end = self.forbidden[-1]
        self.total_cost += self.graph.matrix[end, start]
        self._pheromone_update(end, start)

    def _traditional_best_pick(self):
        #implements equation 2
        np.fill_diagonal(self.graph.pheromone, 0)
        np.fill_diagonal(self.eta_beta, 0)
        return np.multiply(self.graph.pheromone,
                                      self.eta_beta)

    def _udpo_best_pick(self):
        #implement equation 11 for unit distance pheromone operator
        np.fill_diagonal(self.graph.pheromone, 0)
        np.fill_diagonal(self.eta_beta, 0)
        return np.multiply(self.graph.pheromone,
                              self.graph.eta)

    def _pheromone_update(self, current, nexts):
        self._acs_update(current, nexts)


    def _acs_update(self, current, nexts):
        #equation 6 in paper
        #we have a symmetric matrix
        self.pheromone_delta[current, nexts] = (1-self.aco.psi)* \
                                                                 self.graph.pheromone[current, nexts] + \
                                                                 self.aco.psi*self.graph.initial_pheromone
        self.pheromone_delta[nexts, current] = self.pheromone_delta[current, nexts]


class PPACO(object):
    def __init__(self, acs_alpha = 1.0, acs_beta=4.0, acs_rho=0.3, acs_phi = 0.1,
                mmas_alpha = 1.0, mmas_beta = 3.0, mmas_rho = 0.2, dgm_alpha=1.0,
                dgm_beta = 4.0, dgm_rho = 0.4, dgm_phi=0.2, rho_span_min = 1, rho_span_max = 1,
                h0 = 0.9, w0 = 3, q0 = 0.5, k=1, num_ants=300, generations = 100, multiprocess =True):

        '''Here we specify the values for the algorihms acs, mmas and dgm
        alpha = weighting parameter for pheromones
        beta = weighitng parameter for distance-cost
        rho = global pheromone decay/evaporation parameter
        phi = local pheromone decay/evaporation parameter
        rho_span_min = pheromone weighting paramter for the minimal span tree path
        rho_span_max = pheromone weighting paramter for the maximum span tree path
        h0 = threshold for min-span tree similarity
        w0 = threshold for num rounds unchanged min span tree similarity
        q0 = threshold for pribabilisitc ant in acs and dgm
        k = number of knn colonies: must be total_colonies -2
        rule: 1 = acs, 2=mmas, 3=dgmaco
        '''

        random.seed(12345)
        self.acs_alpha = acs_alpha
        self.acs_beta = acs_beta
        self.acs_rho = acs_rho
        self.acs_phi = acs_phi
        self.acs_q0 = q0
        self.mmas_alpha = mmas_alpha
        self.mmas_beta = mmas_beta
        self.mmas_rho = mmas_rho
        self.dgm_alpha = dgm_alpha
        self.dgm_beta = dgm_beta
        self.dgm_rho = dgm_rho
        self.dgm_phi = dgm_phi
        self.rho_span_min = rho_span_min
        self.rho_span_max = rho_span_max
        self.generations = generations
        #eq 13
        self.h0 = h0
        self.w0 = w0
        self.u = {'counter': {}, 'path_index':{}}
        self.k = k

        self.acs = ACO(num_ants, generations, acs_alpha, acs_beta,
                       acs_rho, acs_phi, rho_span_min, rho_span_max, 1, multiprocess=multiprocess)
        self.mma = ACO(num_ants, generations, mmas_alpha, mmas_beta,
                       mmas_rho, 0, rho_span_min, rho_span_max, 2, multiprocess=multiprocess)
        self.dgm = ACO(num_ants, generations, dgm_alpha, dgm_beta,
                       dgm_rho, dgm_phi, rho_span_min, rho_span_max, 3, multiprocess=multiprocess)
        self.colonies = [self.acs, self.mma, self.dgm]


    def solve(self, graph:Graph):
        #set up the solving variables
        self.min_st_inds = [[],[]]
        self.max_st_inds = [[], []]
        self.min_st_inds[0], self.min_st_inds[1] = prim_min.get_minimum_cost(graph.matrix)
        self.max_st_inds[0], self.max_st_inds[1] = prim_max.get_maximum_cost(graph.matrix)
        self.graphs = [copy.deepcopy(graph) for _ in range(len(self.colonies))]
        for colony in self.colonies:
            colony.min_st_inds = copy.deepcopy(self.min_st_inds)
            colony.max_st_inds = copy.deepcopy(self.max_st_inds)
            colony._setup_solver_variables()

        for gen in tqdm(range(self.generations)):
            for i, colony in enumerate(self.colonies):
                colony._sovlve_gen(graph = self.graphs[i], gen = gen)
            self._fix_local_minima()
        i = self._compare_colony_soluations()
        self.cost_tracker = self.colonies[i].cost_tracker
        return self.colonies[i].best_solution, self.colonies[i].best_cost, self.colonies[i].avg_costs, self.colonies[i].best_costs

    def _compare_colony_soluations(self):
        #selects the best solution to return
        best = float('inf')
        ind = -1
        for i, col in enumerate(self.colonies):
            if col.best_cost < best:
                best = col.best_cost
                ind = i
        return ind


    def _fix_local_minima(self):
        self._find_immersion()
        for key in self.u['counter'].keys():
            if self.u['counter'][key] > self.w0:
                simil, entr = self._calc_similarity_entropy(key, self.u['path_index'][key])
                #eq. 14
                sg = max([abs(val) for val in simil.values()])
                max_entr = max(list(entr.values()))
                inv_entr = {v:k for k,v in entr.items()}
                h = inv_entr[max_entr] # colony index for highest entropy
                pearson_phero = self._calc_knn_pheromones(u = key, simil=simil)
                #make s0 matrix
                s0 = np.ones(shape=pearson_phero.shape)
                tril = np.tril_indices_from(s0, k=-1)
                #get s0 random cutoffs in number of lower triangle indices
                s0[tril] = np.random.uniform(0,1, int(s0.shape[0] * (s0.shape[0] - 1) / 2))
                mask = (s0 <= sg)
                self.graphs[key].pheromone[mask] = pearson_phero[mask]
                self.graphs[key].pheromone[~mask] = self.graphs[h].pheromone[~mask]
                #fix upper triangle and diagonal
                self.graphs[key].pheromone.T[tril] = self.graphs[key].pheromone[tril]
                np.fill_diagonal(self.graphs[key].pheromone, 0)


    def _find_immersion(self):
        #implements eq. 13 in art_intell paper
        for i, colony in enumerate(self.colonies):
            for j, solv in enumerate(colony.best_solution):
                c = helper.get_span_tree_simil(solv, self.min_st_inds)
                if c > self.h0:
                    self.u['counter'][i] = self.u['counter'].get(i, 0) + 1
                    self.u['path_index'][i] = j
                    break
                elif colony.best_solution.index(solv) == len(colony.best_solution)-1:
                    self.u['counter'][i] = 0
                    self.u['path_index'][i] = -1

    def _calc_similarity_entropy(self, u, path_index):
        #implements eq.9 and eq. 10 in art_intell paper
        simil = {}
        entropy = {}
        for v, col in enumerate(self.colonies):
            if v != u: #do not comapre to self!
                #get lower traingle and flatten then compare pearson (as it is symmetric matrix)
                #eq. 9
                v_pher = self.graphs[v].pheromone[np.tril_indices_from(self.graphs[v].pheromone, k=-1)] * 10**10
                u_pher = self.graphs[u].pheromone[np.tril_indices_from(self.graphs[u].pheromone, k=-1)] * 10**10
                sim = stats.pearsonr(u_pher, v_pher)[0]
                #eq 10
                entr = stats.entropy(self._get_path_distirb(col))
                simil[v] = sim
                entropy[v] = entr
        return simil, entropy

    def _get_path_distirb(self, aco: ACO):
        #implements path distrib for entropy calculation
        c = {}
        for ant in aco.ants:
            for i in range(1, len(ant.forbidden)):
                path = [ant.forbidden[i-1], ant.forbidden[i]]
                if path[0] > path[1]: #sort so that 4->5 is the same as 5->4 (invalidate direction)
                    path = [path[1], path[0]]
                c['{}{}'.format(path[0], path[1])] = c.get('{}{}'.format(path[0], path[1]), 0) + 1
        total = sum(c.values())
        c = [val/total for val in c.values()]
        return c

    def _calc_knn_pheromones(self, u:int, simil:dict):
        #implements s0<=sg case of eq. 15 of art_intell paper
        simil = {v:k for k,v in simil.items()}
        sscore = sorted(list(simil.keys()))[::-1]

        r_u = np.mean(self.graphs[u].pheromone[np.tril_indices_from(self.graphs[u].pheromone, k=-1)])
        #make tmp pheromone matrix
        tmp = np.zeros(shape=self.graphs[0].pheromone.shape)
        sim_sum = 0
        for i, sc in enumerate(sscore):
            if i < self.k: #check if knn
                r_v = np.mean(self.graphs[simil[sc]].pheromone[np.tril_indices_from(self.graphs[simil[sc]].pheromone, k=-1)])
                tmp = tmp + sc * (self.graphs[simil[sc]].pheromone - r_v)
                sim_sum += sc
        tmp = r_u + tmp/sim_sum
        np.fill_diagonal(tmp, 0)
        return tmp

if __name__ == '__main__':
    # testing
    random.seed(12345)
    cost = np.zeros(shape=(10,10))
    tril = np.tril_indices_from(cost, k=-1)
    cost[tril] = np.arange(len(tril)+1)
    cost.T[tril] = cost[tril]
    graph = Graph(cost_matrix=cost)
    acos = PPACO(multiprocess=True)
    res = acos.solve(graph = copy.deepcopy(graph))
    print('Cost for best solution is {}'.format(res[1]))
