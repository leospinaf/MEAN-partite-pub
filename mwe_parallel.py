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

start = time.time()

## Run the data loading.
expconfig = ExpConfig(
    L=[500,500,500,500,500], U=[500,500,500,500,500], NumEdges=7500, BC=0.1, NumGraphs=10,
    shuffle=True, filename='', seed=1234#42
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
    ComDetMultiCriteria(  # 3D MO approach
    name='3d',
    params = {
        'mode': '3d', # '2d' for 2d approach
        'popsize': 50,
        'termination': None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
        'save_history': False, # set to True for later hypervolume calculations
        'seed': None, # For reproducibility
    }
),
    ComDetMultiCriteria(  # 2D MO approach
    name='2d',
    params = {
        'mode': '2d', # '2d' for 2d approach
        'popsize': 50,
        'termination': None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
        'save_history': False, # set to True for later hypervolume calculations
        'seed': None, # For reproducibility
    }
)
]

print('Algos defined in %f s' % (time.time()-start))
start = time.time()

from moo.communities import run_parallel_communities

start = time.time()
parallelResults = run_parallel_communities(datagen, algos, n_jobs = 6)

parallelTime = time.time()-start
print("Parallel time taken", parallelTime)

df_contestants = pd.DataFrame(parallelResults)
best_solutions = get_best_community_solutions(df_contestants).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15,8))
ax, stats = draw_best_community_solutions(best_solutions,ax)

code.interact(local=locals())