## This script is a minimum working example of the data process.

import igraph
import pandas as pd
from moo.data_generation import ExpConfig, DataGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
import moo.contestant as contestant
from moo.multicriteria import ComDetMultiCriteria
import matplotlib.pyplot as plt
import sknetwork
import code
import time
from copy import deepcopy
import pymoo.util.termination.f_tol

start = time.time()

## Run the data loading.
expconfig = ExpConfig(
    L=[50,50], U=[50,50], NumEdges=200, BC=0.1, NumGraphs=1,
    shuffle=True, filename='test_graphs_', seed=24#42
)

print(expconfig) # Print parameters, or access individually, e.g., expconfig.NumEdges

print('Config defined in %f s' % (time.time()-start))
start = time.time()

# Generate data following the defined experiment confguration
expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters
print(expgen)
datagen = expgen.generate_data() # datagen is an iterator

print('Data generator constructed in %f s' % (time.time()-start))
start = time.time()

## Define the algorithms.
algos = [
    contestant.ComDetMultiLevel(), # Multi-Level approach
    contestant.ComDetEdgeBetweenness(), # EdgeBetweenness approach
    contestant.ComDetWalkTrap(), # WalkTrap approach
    contestant.ComDetFastGreedy(), # FastGreedy approach
    contestant.ComDetBRIM(), # BRIM approach
    contestant.ComDetBiLouvain(), # sknetwork bilovain
    #contestant.ComDetariadne(), # pymocd ariadne
    #contestant.ComDethpmocd(), # pymocd hpmocd
    #contestant.ComDetmocd_q(), # pymocd mocd_q
    #contestant.ComDetmocd_d(), # pymocd mocd_d
    #contestant.ComDetmoga_net(), # pymocd moga_net (Pizzuti)
    #contestant.ComDetccm(), # pymocd ccm
    #contestant.ComDetkrm(), # pymocd krm
    #contestant.ComDetmmcomo(), # pymocd mmcomo
    #ComDetMultiCriteria(  # 3D MO approach
    #name='3d',
    #params = {
    #    'mode': '3d', # '2d' for 2d approach
    #    'popsize': 50,
    #    'termination': None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
    #    'save_history': False, # set to True for later hypervolume calculations
    #    'seed': None, # For reproducibility
    #    'initialization': 'original',
    #    'mutation': '',#'enhanced',
    #    'enhance':0,
    #}
#),
#    ComDetMultiCriteria(  # 2D MO approach
 #   name='2d',
  #  params = {
   #     'mode': '2d', # '2d' for 2d approach
    #    'popsize': 50,
     #   'termination': pymoo.util.termination.f_tol.MultiObjectiveSpaceToleranceTermination(tol=0.0001,n_last=100,nth_gen=75,n_max_gen=5000,n_max_evals=None),#None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
      #  'save_history': False, # set to True for later hypervolume calculations
       # 'seed': None, # For reproducibility
        #'initialization': 'original',
        #'mutation': 'enhanced',#'enhanced',
        #'enhance':0.2,
        #'pair_crossover':1.0,
        #'gene_crossover':0.1
    #}
#)
]

print('Algos defined in %f s' % (time.time()-start))
start = time.time()

## Detect some communities.
results = [] # Holds results of contestants
for g_idx, graph in enumerate(datagen):
    print(f'Processing Graph {g_idx+1}')
    
    ## Generate badj to pass and save computation time.
    #code.interact(local=locals())
    for algo in algos:
        print(f'\tUsing algoithm {algo.name_}')
        result = algo.detect_communities(graph=deepcopy(graph)).get_results()
        # Result is a list of dictionaries, each dictionary stores the metrics of one iteration (see code for details)
        for r in result: # Appending graph index to results, for debugging purposes
            r['graph_idx'] = g_idx + 1
        results.extend(result)
        print('Results collected in %f s' % (time.time()-start))
        start = time.time()

df_contestants = pd.DataFrame(results)
best_solutions = get_best_community_solutions(df_contestants).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15,8))
ax, stats = draw_best_community_solutions(best_solutions,ax)

code.interact(local=locals())