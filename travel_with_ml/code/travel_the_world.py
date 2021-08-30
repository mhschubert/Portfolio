# script to call and run genetic algorithm to evaluate the TSP problem for all world capitals
# author: Marcel H. Schubert

import os
import sys
import argparse
import ast
sys.path.extend(['./'])
import genetic, aco
import copy
from utils import process_data, plot_helpers, helper
import numpy as np
import pickle



def tsp_solve(args:dict):
    dat = process_data.Data(dict['path'])
    dat.clean_data(datacols=['CapitalLatitude', 'CapitalLongitude'])
    dat.make_dist_matrix(longcol='CapitalLatitude', latcol='CapitalLongitude')

    #create graph
    graph = aco.Graph(cost_matrix=dat.matrix)

    #solve with genetic
    gen = genetic.Genetic(graph=copy.deepcopy(graph),
                              mutation_rate=args['mutation-rate'],
                              population_size=args['population-size'],
                              num_gen=args['epochs'],
                              elite_size=args['elite-size'],
                              crossover=args['crossover'])
    res_gen = gen.genetic_solve()

    antcols = aco.PPACO(acs_alpha=args['acs-alpha'], acs_beta=args['acs-beta'],
                        acs_phi=args['acs-phi'], acs_rho=args['acs-rho'],
                        mmas_alpha=args['mms-alpha'], mmas_beta=args['mms-beta'],
                        mmas_rho=args['mms-rho'], dgm_alpha=args['dgm-alpha'],
                        dgm_beta=args['dgm-beta'], dgm_phi=args['dgm-phi'],
                        dgm_rho=args['dgm-rho'], rho_span_max=args['max-span'],
                        rho_span_min=args['min-span'], h0=args['h0'], w0=args['w0'],
                        num_ants=args['population-size'], generations=args['epochs'],
                        multiprocess=args['multiprocess']
                        )

    if not args['PRACO']:
        antcols = copy.deepcopy(antcols.colonies[args['aco_rule']])
    res_aco = antcols.solve(graph=copy.deepcopy(graph))
    os.makedirs('../models', exist_ok=True)
    with open('../models/ant_object.p', 'wb') as w:
        pickle.dump(antcols, w)
    with open('../models/genetic_object.p', 'wb') as w:
        pickle.dump(gen, w)


    if res_gen[0] > res_aco[1]:
        fin = res_aco[0][0]
    else:
        fin = res_gen[1]

    #plot costs over generations
    plot_helpers.plot_costs(np.max(np.array(antcols.cost_tracker), axis=1), algo='Ant', figdir=args['savefig'])
    plot_helpers.plot_costs(np.array(gen.change), algo='Genetic', figdir=args['savefig'])

    #make gifs
    plot_helpers.plot_globes(route=fin, df=dat.df_clean, figdir=args['savefig'], fancypath=args['fancypath'])
    print('"solved" TSP problem for your world-trip...')

if __name__ == '__main__':

    # set up list of arguements
    argparser = argparse.ArgumentParser(description='Arguements for the Genetic Algorithm')
    argparser.add_argument_group('Genetic Algorithm')
    argparser.add_argument('-m', '--mutation-rate', help='GA mutation rate', default='0.02')
    argparser.add_argument('-p', '--population-size', help='GA population size / ACO no. ants', default='200')
    argparser.add_argument('-e', '--epochs', help='No. of epochs', default='200')
    argparser.add_argument('-s', '--elite-size', help='Fraction of elite to keep in GA', default='0.2')
    argparser.add_argument('-c', '--crossover', help='No. of epochs', default='cx1', choices=['ordered', 'cx1', 'cx2', 'random'])

    argparser = argparse.ArgumentParser(description='Arguements for the Ant Colony Algorithm')
    argparser.add_argument_group('ACO Algorithm')
    argparser.add_argument('--acs-alpha', help='ACS alpha', default='1.0')
    argparser.add_argument('--mms-alpha', help='MMS alpha', default='1.0')
    argparser.add_argument('--dgm-alpha', help='DGMACO alpha', default='1.0')
    argparser.add_argument('--acs-beta', help='ACS beta', default='4.0')
    argparser.add_argument('--mms-beta', help='MMS beta', default='3.0')
    argparser.add_argument('--dgm-beta', help='DGMACO beta', default='4.0')
    argparser.add_argument('--acs-phi', help='ACS Phi/PSi', default='0.1')
    argparser.add_argument('--dgm-phi', help='DGMACO Phi/PSi', default='0.2')
    argparser.add_argument('--acs-rho', help='ACS Rho (Pheromone decay)', default='0.3')
    argparser.add_argument('--mms-rho', help='MMS Rho (Pheromone decay)', default='0.2')
    argparser.add_argument('--dgm-rho', help='DGMACO Rho (Pheromone decay)', default='0.4')
    argparser.add_argument('--min-span', help='MMS decay for min-span tree route impact', default='1.0')
    argparser.add_argument('--max-span', help='MMS decay for max-span tree route impact', default='1.0')

    #path for data
    argparser = argparse.ArgumentParser(description='Data and Save Directories, Multiprocessing')
    argparser.add_argument_group('dirinfo')
    argparser.add_argument('--datapath', help='Relative path to dataset', default='../data')
    argparser.add_argument('--savepath', help='Relative path to figure directory', default='../figures')
    argparser.add_argument('--multiprocess', help='ACO algorithm with multiprocessing to speed things up?', default='False',
                           choices=['True', 'False'])

    #parse arguements
    args = vars(argparser.parse_args())
    #convert to types
    for key in args.keys():
        if key not in ['datapath', 'savepath']:
            args[key] = ast.literal_eval(args[key])
        else:
            # fix windows-unix path problem
            args[key] = os.path.join(*helper.split_path_unix_win(args[key]))

    args['w0'] = 3
    args['h0'] = 0.9

    tsp_solve(args)

    print('done')





