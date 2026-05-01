# Introduction
This repository contains code related to the paper “Multi-objective community detection for bipartite graphs”.

## File/directory description and usage
- `moo` directory contains the core functionality of the Meanpartite method. It contains the updated code for data generation (`data_generation.py`), contestant algorithms (`contestant.py`), multicriteria approach (`multicriteria.py`) and a utility module (`utils.py`) which provides functionality for writing/reading graphs into various file formats, writing graphs and reading graphs from the format used in the legacy code, etc. Several methods for solving the community detection problem are included in `communities.py` and core functionality from skbio is reproduced in `skbio_gini.py` for compatibility purposes.

- `plot_figures.py` contains the information required to reproduce many of the network figures included in the paper. The outputs from this script are included in the `figs` directory.

- The list of package and version requirements for the library are included in `requirements.txt`. A virtual environment can be configured with this setup using `pip install -r requirements.txt`. This codebase has been tested on Python 3.xx.

## Parameter descriptions
The synthetic data generation processes uses a number of parameters to control properties of the bipartite networks and their communities. Parameter defaults and options are explained below.
- `L`: list of number of lower nodes in each community. Defaults to `[40,60]`.
- `U`: list of number of upper nodes in each community. Defaults to `[40,60]`.
- `NumEdges`: number of edges to include in simulated graph. Note that due to trimming to the giant component, this is a maximum value and not guaranteed. Defaults to `200`.
- `BC`: the probability of a generated edge connecting nodes in different communities. Defaults to `0.2`.
- `NumGraphs`: the number of graphs to generate using this set of parameters. Defaults to `30`.
- `shuffle`: Boolean for shuffling nodes labels to remove relationship between node label and community label. Defaults to `True`.
- `filename`: provide a custom filename structure to be used alongside graph number by generator. Defaults to the empty string, returning filenames `_0.gml`, `_1.gml` etc.
- `seed`: define a random seed for reproducibility. Defaults to `42`.

The community detection process uses a number of parameters to change the optimization process. Parameter defaults and options are explained below.
- `name`: provide an informative string to differentiate different sets of initialization parameters. Defaults to `multicriteria`.
- `params`: dictionary of named parameters determining algorithm behaviour. Possible inclusions are detailed below.
    - `mode`: determine the number of objectives to consider. Defaults to `3d` (modularity of projections and number of communities), alternatives are `2d` (modularity of original graph and number of clusters) or `4d` (modularity of both projections and original graph and number of communities).
    - `popsize`: the population of solutions to consider during each iteration. Defaults to `50`.
    - `termination`: the termination criteria for the optimization process. Defaults to after 1000 generations, otherwise pass any pymoo termination object.
    - `save_history`: determine whether to save information about each optimization step, required to calculate hypervolumes. Defaults to `True`.
    - `seed`: define a random seed for reproducibility. Defaults to random selection, otherwise pass an integer.
    - `initializaton`: define how the starting population should be generated. Defaults to `variant` (identify different numbers of communities and use minimum spanning trees), alternatives are `pizzuti` (sample random edges from the graph) or `original` (add diversity to the minimum spanning tree).
    - `mutation`: define which mutation operator should be used. Defaults to `enhanced` (mutate each node with probability dependent on degree and centrality, with consideration for the projected edges), alternatives are `pizzuti` (uniform integer mutation for each design variable), `int_pm` (`get_mutation(int_pm)` from pymoo) or the empty string (mutate each node with probability dependent on degree and centrality).
    - `enhance`: define the proportion of enhacement in the bipartite graph using the projected edges. Defaults to `0` (no consideration of projected edges), wherein `enhanced` and empty string for mutation are the same.
    - `min_num_clusters`: define the minimum number of communities returned by the algorithm. Defaults to `1`.
    - `max_num_clusters`: define the maximum number of communities returned by the algorithm. Defaults to `30`.
    - `pair_crossover`: define the probability for each population pair to be considered for crossover. Defaults to `1`.
    - `gene_crossover`: define the probability (or distribution) for each gene to be switched during crossover. If passing a distribution, this parameter expects a comma-separated string of `numpy.random` function and (non-size) parameters to be set in order. Defaults to `0.5`.
