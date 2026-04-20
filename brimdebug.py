# BRIM explore
# This is working with version 1.1 of Condor - git commit 389932


import igraph
from moo.data_generation import ExpConfig, DataGenerator
from moo.data_generation import ExpConfig, DataGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
from moo.communities import run_parallel_communities
import moo.contestant as contestant
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


import pandas as pd


expconfig = ExpConfig(
    L=[15,15], U=[15,15],
    NumEdges=200,
    BC=0.1, NumGraphs=30,
    shuffle=True,
    seed=1234  
    )



algos = [

    contestant.ComDetFastGreedy(), # FastGreedy approach
    contestant.ComDetBRIM(),
]

expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters

datagenSeries = expgen.generate_data() 


seriesResults = [] # Holds results of contestants
for g_idx, graph in enumerate(datagenSeries):
#    print(f'Processing Graph {g_idx+1}')
    for algo in algos:
#        print(f'\tUsing algoithm {algo.name_}')
        result = algo.detect_communities(graph=graph).get_results()
        # Result is a list of dictionaries, each dictionary stores the metrics of one iteration (see code for details)
        for r in result: # Appending graph index to results, for debugging purposes
            r['graph_idx'] = g_idx + 1
        seriesResults.extend(result)

df_seriesResults = pd.DataFrame(seriesResults)


# Extract best solutions for each graph/algorithm pair

best_solutions = get_best_community_solutions(df_seriesResults)

print(best_solutions)
