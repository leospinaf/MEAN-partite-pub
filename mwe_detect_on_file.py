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

from moo.communities import run_communities_from_file
import pymoo.util.termination.f_tol as pymoo_termination

## Define the algorithm.
algo = ComDetMultiCriteria(name = "2d",
						  params = {
							'mode': '2d', # '2d' for 2d approach
							'popsize': 50,
							'termination': pymoo_termination.MultiObjectiveSpaceToleranceTermination(tol = 0.0001,
																									 n_last = 100,
																									 nth_gen = 75,
																									 n_max_gen = 5000,
																									 n_max_evals = None),
							'save_history': False, # set to True for later hypervolume calculations
							'seed': None, # For reproducibility
							'initialization': 'variant',
							'mutation': 'enhanced',
							'enhance' : 0.1
						})


## Detect the communities on the graph.
results = pd.DataFrame(run_communities_from_file('errors_to_test/Experiment_500_5_3_1_0.1_1_Symmetric_2_0.gml',algo))

code.interact(local=locals())