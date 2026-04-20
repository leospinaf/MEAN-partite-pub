library(reticulate)
library(igraph)
library(tidyverse)


use_virtualenv("./venv")

moo <- import("moo")
moo_datagen <- import("moo.data_generation")
contestant <- import("moo.contestant")
moo_communities <- import("moo.communities")
multicriteria <- import("moo.multicriteria")
pd <- import("pandas")


expconfig <- moo_datagen$ExpConfig(
  L=list(500L,500L,500L,500L,500L), U=list(500L,500L,500L,500L,500L),
  NumEdges=7500L, 
  BC=0.1, NumGraphs=100L, # Was 30L
  shuffle=TRUE, 
  seed=1234L  
  
)

expconfig <- moo_datagen$ExpConfig(
  L=list(150L,150L), U=list(150L,150L),
  NumEdges=200L, 
  BC=0.1, NumGraphs=1L, # Was 30L
  shuffle=TRUE, 
  seed=24L  
  
)

algos = list(
  contestant$ComDetMultiLevel(), # Multi-Level approach
  #contestant$ComDetEdgeBetweenness(), # EdgeBetweenness approach
  #contestant$ComDetWalkTrap(), # WalkTrap approach
  #contestant$ComDetFastGreedy(), # FastGreedy approach
  contestant$ComDetBRIM(max_num_clusters=1L)#, # Brim
  #multicriteria$ComDetMultiCriteria(name = "3d",
  #                                  params = list(
  #                                    'mode'= '3d', # '2d' for 2d approach
  #                                    'popsize'= 50L,
  #                                    'termination'= py_none(), # By default it runs for 1000 generations (or pass a pymoo termination instance)
  #                                    'save_history'= FALSE, # set to True for later hypervolume calculations
  #                                    'seed'= py_none() # For reproducibility
  #                                  ))
)

## The error from Luis is as below.
moo_multicriteria <- import("moo.multicriteria")
pymoo_termination <- import("pymoo.util.termination.f_tol")
algos_2d_newmut_enh = list(moo_multicriteria$ComDetMultiCriteria(name="2d_newmut_enh",
			params = list(
			'mode'='2d',
			'popsize'=50L,
			'termination'=pymoo_termination$MultiObjectiveSpaceToleranceTermination(tol=0.0001,
									n_last=100L,
									nth_gen=75L,
									n_max_gen=5000L,
									n_max_evals=py_none()),
			'save_history'=FALSE,
			'seed'=py_none(),
			'initialization'='original',
			'mutation'='enhanced',
			'enhance'=0.2,
			'pair_crossover'=1,
			'gene_crossover'=0.1
)) )


expgen = moo_datagen$DataGenerator(expconfig=expconfig) # Pass defined parameters

datagen <- expgen$generate_data()

results <- moo_communities$run_serial_communities(datagen, algos_2d_newmut_enh)
results <- moo_communities$run_parallel_communities(datagen, algos)

# I've been unable to get parallel processing working on Windows.  It's fine
# on Windows Subsystem for Linux (WSL).  I strongly suspect it'll be fine on 
# "proper" Linux and macos.  
# Have tried using fix at https://github.com/rstudio/reticulate/issues/517
# results <- moo_communities$run_parallel_communities(datagen, algos, n_jobs=2)

results_df = pd$DataFrame(results)

best_solutions <- contestant$get_best_community_solutions(results_df)


best_solutions%>% as_tibble() %>% 
  ggplot(aes(x = name, y = adj_rand_index)) + 
  geom_boxplot()

## Get the convergence plot for the first algo.
alg = algos[1]
vol_dat = alg$compute_hypervolume()
