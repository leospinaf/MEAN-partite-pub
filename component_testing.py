import moo.multicriteria
import pymoo.util.termination.f_tol
import code
from pymoo.factory import get_termination
import numpy as np
import functools
import numpy as np

## Generate a dummy graph for testing purposes.
from moo.data_generation import ExpConfig, DataGenerator
expconfig = ExpConfig(
	L=[150,150], U=[150,150], NumEdges=500, BC=0.1, NumGraphs=1,
	shuffle=True, filename='test_graphs_', seed=24#42
)

expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters
datagen = expgen.generate_data() # datagen is an iterator
graph = next(datagen)

## Test the mutation probabilities calculated during problem initialisation.
## Looking in algo.problem_.mut_list, those elements != 1 are rounding errors (differ only at 16 dp).
algo = moo.multicriteria.ComDetMultiCriteria(  # 2D MO approach
	name='2d',
	params = {
		'mode': '2d', # '2d' for 2d approach
		'popsize': 50,
		'termination': get_termination("n_gen", 2),  ## At least 2 is required to get any iterations (gen 1 is the inital pop).
		'save_history': True, # set to True for later hypervolume calculations
		'seed': 42, # For reproducibility
		'initialization': 'variant',#'original',
		'mutation': 'disable',#'enhanced',
		'enhance':0,
		'pair_crossover':1.0,
		'gene_crossover':{'dist':'beta','a':0.077,'b':0.693}##functools.partial(np.random.beta,a=0.077,b=0.693)#0.1
	}
)

algo.detect_communities(graph)

## Test the crossover function on one step to see whether probabilities are applied correctly.
## Want to compare the initial population and final population, ideally without any mutation, just crossover.
## The algo object includes the population at each step if the history is saved.
## algo.res_.history[0].pop[0].get('X') returns the genome for the first member of the population at time 0 etc.
def get_pop_at_step(algo,i):
	""" Return the population array of the given algorithm as step i. """
	
	genomes = []
	for ind in algo.res_.history[i].pop:
		genomes.append(ind.get('X'))
	
	return np.vstack(genomes)

def get_final_pop(algo):
	""" Return the population array after the algorithm has terminated. """
	
	genomes = []
	for ind in algo.res_.pop:
		genomes.append(ind.get('X'))
	
	return np.vstack(genomes)

code.interact(local=locals())