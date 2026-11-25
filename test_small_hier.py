# This script uses the hierarchical generator to test a small graph and the derivation of the ARI statistics.

import igraph
import pandas as pd
from moo.data_generation import ExpConfig, DataGenerator, HierarchicalConfig, HierarchicalGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
import moo.contestant as contestant
from moo.multicriteria import ComDetMultiCriteria
import matplotlib.pyplot as plt
import sknetwork
import code
import time
from copy import deepcopy
from pymoo.termination.default import DefaultMultiObjectiveTermination
import os
from sklearn.metrics import adjusted_rand_score
import numpy as np
import networkx as nx

start = time.time()

print('Generating new graphs')
## Run the data loading.
#expconfig = ExpConfig(
#    L=[50,50], U=[50,50], NumEdges=200, BC=0.1, NumGraphs=1,
#    shuffle=True, filename='test_graphs_', seed=24#42
#)
expconfig = HierarchicalConfig(
	L=[5,5,5,5,5], U=[5,5,5,5], NumEdges=50, ProbArray=[0.7,0.2,0.05,0.05,0.2,0.7,0.05,0.05,0.2,0.2,0.2,0.2,0.05,0.05,0.7,0.2,0.05,0.05,0.2,0.7],
	HighGT=[1,1,2,2,2,1,1,2,2],filename='hier_test_',seed=42,NumGraphs=1
)

print(expconfig) # Print parameters, or access individually, e.g., expconfig.NumEdges

print('Config defined in %f s' % (time.time()-start))
start = time.time()

# Generate data following the defined experiment confguration
#expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters
expgen = HierarchicalGenerator(expconfig=expconfig)
print(expgen)
datagen = expgen.generate_data() # datagen is an iterator

print('Data generator constructed in %f s' % (time.time()-start))
start = time.time()

## Define the algorithms.
algos = [
    contestant.ComDetMultiLevel(return_membership=True), # Multi-Level approach
    contestant.ComDetEdgeBetweenness(return_membership=True), # EdgeBetweenness approach
    contestant.ComDetWalkTrap(return_membership=True), # WalkTrap approach
    contestant.ComDetFastGreedy(return_membership=True), # FastGreedy approach
    contestant.ComDetBRIM(return_membership=True), # BRIM approach
    contestant.ComDetBiLouvain(return_membership=True), # sknetwork bilovain
    contestant.ComDetscale(return_membership=True), # pymocd scale, using the full Pareto front
    contestant.ComDethpmocd(return_membership=True), # pymocd hpmocd
    contestant.ComDetmocd_q(return_membership=True), # pymocd mocd_q
    contestant.ComDetmocd_d(return_membership=True), # pymocd mocd_d
    contestant.ComDetmoga_net(return_membership=True), # pymocd moga_net (Pizzuti)
    contestant.ComDetccm(return_membership=True), # pymocd ccm
    contestant.ComDetkrm(return_membership=True), # pymocd krm
    contestant.ComDetmmcomo(return_membership=True), # pymocd mmcomo, using the full Pareto front
    contestant.ComDetSBM(return_membership=True), # sparsebm SBM
    contestant.ComDetLBM(return_membership=True), # sparsebm LBM
    ComDetMultiCriteria(  # 3D MO approach
    name='3d',
    params = {
        'mode': '3d', # '2d' for 2d approach
        'popsize': 50,
        'termination': None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
        'save_history': False, # set to True for later hypervolume calculations
        'seed': None, # For reproducibility
        'initialization': 'original',
        'mutation': '',#'enhanced',
        'enhance':0,
        'return_membership':True,
    }
),
    ComDetMultiCriteria(  # 2D MO approach
    name='2d',
    params = {
        'mode': '2d', # '2d' for 2d approach
        'popsize': 50,
        'termination': DefaultMultiObjectiveTermination(ftol=1e-4,period=100,n_max_gen=5000,n_max_evals=None,n_skip=75),#None, # By default it runs for 1000 generations (or pass a pymoo termination instance)
        'save_history': False, # set to True for later hypervolume calculations
        'seed': None, # For reproducibility
        'initialization': 'original',
        'mutation': 'enhanced',#'enhanced',
        'enhance':0.2,
        'pair_crossover':1.0,
        'gene_crossover':0.1,
        'return_membership':True
    },
)
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

#fig, ax = plt.subplots(figsize=(15,8))
#ax, stats = draw_best_community_solutions(best_solutions,ax)

# Calculate the ARI at the higher level for each solution.
gt = graph.vs['HGT']
df_contestants['hier_ari_check'] = [adjusted_rand_score(gt,comms) if comms != 'Not returned' else np.nan for comms in df_contestants['membership']]

# --- settings ---------------------------------------------------------------
COLOUR_ATTRS = ("GT", "HGT", "membership")  # one per panel
TYPE_ATTR = "type"
LAYOUT = "spring"                 # "bipartite" or "spring"
NODE_SIZE = 120
CMAP = "viridis"                     # numeric attributes
CAT_CMAP = "tab10"                   # categorical attributes
# ----------------------------------------------------------------------------

SHAPES = ["o", "s", "^", "D", "v", "p", "h", "*"]

from plot_bipartite import *

# Convert to nx for plotting
G = graph.to_networkx()

types = sorted({d.get(TYPE_ATTR) for _, d in G.nodes(data=True)}, key=str)
shape_map = {t: SHAPES[i % len(SHAPES)] for i, t in enumerate(types)}
pos = compute_layout(G, TYPE_ATTR, LAYOUT)  # shared across panels

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, attr in zip(axes, COLOUR_ATTRS):
    draw_panel(ax, G, pos, attr, TYPE_ATTR, shape_map)
fig.tight_layout()
plt.savefig('lower_upper_gt_colours.png',bbox_inches='tight')
plt.close()

max_ari = {}  # method: max ari
for id, sol in df_contestants.iterrows():
    if max_ari.get(sol['name']) == None:
        plot_it = True

    elif sol['adj_rand_index'] > max_ari[sol['name']]:
        plot_it = True
    else:
        plot_it = False
    
    if plot_it:

        max_ari[sol['name']] = sol['adj_rand_index']

        nx.set_node_attributes(G, dict(zip(G.nodes, sol['membership'])), "membership")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        for ax, attr in zip(axes, COLOUR_ATTRS):
            print(f'plotting {attr}')
            draw_panel(ax, G, pos, attr, TYPE_ATTR, shape_map)
        fig.tight_layout()
        plt.savefig(f'best_{sol["name"]}.png',bbox_inches='tight')
        plt.close()


code.interact(local=locals())