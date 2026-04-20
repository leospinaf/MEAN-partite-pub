from moo.data_generation import ExpConfig, DataGenerator

from joblib import Parallel, delayed
import itertools

import pandas as pd
from tqdm import tqdm
import igraph


# A utility function to generate data for each configuration, and run the community detection algorithms
import pandas as pd
def detect_communitites(expconfig, algos):
    '''
    Generates data as per expconfig parameters and runs community derection algorithms
    '''

    # Generate data
    expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters
    print(expgen)
    datagen = expgen.generate_data() # datagen is an iterator

    results = [] # Holds results of community detection algorithms (list of dictionaries)
    for g_idx, graph in enumerate(datagen):
        # if g_idx >= 1: #num_graphs_to_run:
        #     break
        # else:
        #print(f'Processing Graph {g_idx+1}')
        for algo in algos:
            #print(f'  Using algoithm {algo.name_} ... ', end='')
            result = algo.detect_communities(graph=graph).get_results()
            # Result is a list of dictionaries, each dictionary stores the metrics of one iteration (see code for details)
            #print(f'Done')
            for r in result: # Appending graph index to results, for debugging purposes
                r['graph_idx'] = g_idx + 1
            results.extend(result)
    return results


def run_parallel_communities(graphgenerator, algos, n_jobs = 4):
    '''
    Run all combinations of the the graphs produced by the graphgenerator iterator and
    algorithms specified in the algos list, over n_jobs jobs in parallel

    graphgenerator An iterator produced by the generate_data() method applied to an expconfig
    algos A list containing the algorithms to apply to each graph

    Returns a list containing the results of each step of the algorithm.  You will likely want 
    to select the best solution for each graph / algorithm combination

    '''

    # Get all combinations of graphs and algorithms
    combinedList = itertools.product(enumerate(graphgenerator), algos)

    def runalgo(c):
        '''
        Wrapper function to get the graph object, 
        '''
        # Extract elements we need
        ig, algo = c
        i, g  = ig
        #print(i, algo)
        result = algo.detect_communities(graph=g).get_results()
        for r in result: 
            r['graph_idx'] = i + 1

        return(result)



    joblibresultsStacked = Parallel(n_jobs = n_jobs) (delayed(runalgo)(c) for c in tqdm(combinedList))

    # Unnest the list we get
    def flatten(t):
        return [item for sublist in t for item in sublist]

    joblibresults = flatten(joblibresultsStacked)

    return joblibresults


def run_serial_communities(graphgenerator, algos):
    '''
    Run all combinations of the the graphs produced by the graphgenerator iterator and
    algorithms specified in the algos list, in series

    graphgenerator An iterator produced by the generate_data() method applied to an expconfig
    algos A list containing the algorithms to apply to each graph

    Returns a list containing the results of each step of the algorithm.  You will likely want 
    to select the best solution for each graph / algorithm combination

    '''

    # Get all combinations of graphs and algorithms
    combinedList = itertools.product(enumerate(graphgenerator), algos)

    def runalgo(c):
        '''
        Wrapper function to get the graph object, 
        '''
        # Extract elements we need
        ig, algo = c
        i, g  = ig
        #print(i, algo)
        result = algo.detect_communities(graph=g).get_results()
        for r in result: 
            r['graph_idx'] = i + 1

        return(result)


    joblibresultsStacked = map(runalgo, combinedList)
    
    # Unnest the list we get
    def flatten(t):
        return [item for sublist in t for item in sublist]

    joblibresults = flatten(joblibresultsStacked)

    return joblibresults

def run_communities_from_file(fname, algo):
    '''
    Load the specified graph from file and detect communities using the given algorithm.

    fname A string giving the path and filename for the graph in question.

    Returns a list containing the results of each step of the algorithm.  You will likely want 
    to select the best solution for each graph / algorithm combination

    '''

    ## Read in the graph from file.
    graph = igraph.Graph.Read_GML(fname)

    results = algo.detect_communities(graph).get_results()

    for r in results:
        r['graph_idx'] = fname

    return results
