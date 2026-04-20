# Introduction
- This repository contains code related to the paper “Multi-objective community detection for bipartite graphs” (see untouched/BipartiteCommunityPremium(2).pdf)
- A Research IT work has been conducted to test the code, clean and rewrite it in a way to be shared with the community.

## File/directory description and usage
- ‘untouched’ directory contains the conference paper versions, other literature on the topic, and all original (unaltered) notebooks which were used to generate the paper experiments (non-shuffled versions, see below). Please note that they are not working as they use previous versions of pymoo, condor, and igraph packages (igraph 0.9.1, pymoo 0.4.2.2, condor 1.1). They mainly serve documentation and archival purposes. For running versions of the 3 notebooks (data generation, contestants, and multicriteria approach), please see below.

- Notebooks 1. Data Generation.ipynb and Data Generation (shuffled).ipynb, 2. Contestant.ipynb and Contestant (shuffled).ipynb, 3. Multicriterion approach.ipynb are the working notebooks adapted from the aforementioned ones (the shuffled tag denotes notebooks which shuffle graph vertex indices during data generation, they are not used in the paper results). The code remains the same, the modifications include updating them to work with the current package versions (igraph 0.9.9, pymoo 0.5.0, condor 2.0). Please note that condor package has been modified to be able to run BRIM algorithm properly, the updated condor code is include within this repository (see below).

- Files multicriterion_3d.py and multicriterion_2d.py are similar to Multicriterion approach.ipynb but they allow running the legacy code in 3d or 2d modes.

- ‘condor’ directory contains the updated condor package to make the code run. If one is interested in using the old version fo condor, one needs to update the code for the older condor interface and import/use condor_1.1.py file (included) instead.

- ‘moo’ directory is the actual code package that replaces the legacy code. It contains the updated code for data generation (data_generation.py), contestant algorithms (contestant.py), multicriteria approach (multicriteria.py) and a utility module (utils.py) which provides functionality for writing/reading graphs into various file formats, writing graphs and reading graphs from the format used in the legacy code, etc. The usage of the new code (package moo) is explained by example in the notebooks (see below)

- Notebooks 01_Data Generation.ipynb 02_Contestants.ipynb 03_Multicriteria Approach.ipynb paper_figures.ipynb show many examples of how to use the package. Their usage is recommended.

## Notes
- The code can be used as it is to reproduce the paper results (see notebooks 02_Contestant.ipynb, 03_Multicritera Approach.ipynb, and paper_figures.ipynb). It can also be used with 3rd party data (graphs and groud truth data separated) by following the example in notebook 01_Data Generation.ipynb.
- Legacy code does not guarantee reproducibility. Please use the new code instead (allows passing a random number generator seed).
- Pending work consists in adapting the code to return communities as results, in the case of missing ground truth.


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