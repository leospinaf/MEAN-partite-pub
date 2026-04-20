
import igraph
from moo.data_generation import ExpConfig, DataGenerator
from moo.data_generation import ExpConfig, DataGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
from moo.communities import run_parallel_communities
import moo.contestant as contestant
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


import pandas as pd

import time

expconfig = ExpConfig(
    L=[40,60], U=[200,300],
    NumEdges=1000,
    BC=0.1, NumGraphs=30,
    shuffle=True, 
    seed=1234  
    )


algos = [
    #contestant.ComDetMultiLevel(), # Multi-Level approach
    contestant.ComDetEdgeBetweenness(), # EdgeBetweenness approach
    contestant.ComDetWalkTrap(), # WalkTrap approach
    contestant.ComDetFastGreedy(), # FastGreedy approach
]


expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters

datagenSeries = expgen.generate_data() 
datagenParallel = expgen.generate_data() 

start = time.time()
parallelResults = run_parallel_communities(datagenParallel, algos, n_jobs = 7)

parallelTime = time.time()-start
print("Parallel time taken", parallelTime)

# start = time.time()
# seriesResults = [] # Holds results of contestants
# for g_idx, graph in enumerate(datagenSeries):
# #    print(f'Processing Graph {g_idx+1}')
#     for algo in algos:
# #        print(f'\tUsing algoithm {algo.name_}')
#         result = algo.detect_communities(graph=graph).get_results()
#         # Result is a list of dictionaries, each dictionary stores the metrics of one iteration (see code for details)
#         for r in result: # Appending graph index to results, for debugging purposes
#             r['graph_idx'] = g_idx + 1
#         seriesResults.extend(result)

# seriesTime = time.time()-start
# print("Series time taken", seriesTime)



# print("Speedup:", seriesTime/parallelTime)
# df_Parallel = pd.DataFrame(parallelResults)
# df_Series = pd.DataFrame(seriesResults)

# print(df_Parallel.equals(df_Series))





