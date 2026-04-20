library(reticulate)


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
  L=list(50L,50L), U=list(50L,50L),
  NumEdges=1000L,
  BC=0.2, NumGraphs=5L, # Was 30L
  shuffle=TRUE, 
  seed=1234L  
  
)

algos = list(
  contestant$ComDetMultiLevel(), # Multi-Level approach
  contestant$ComDetEdgeBetweenness(), # EdgeBetweenness approach
  contestant$ComDetWalkTrap(), # WalkTrap approach
  contestant$ComDetFastGreedy(), # FastGreedy approach
  contestant$ComDetBRIM(), # Brim
  multicriteria$ComDetMultiCriteria(name = "3d",
                                    params = list(
                                      'mode'= '3d', # '2d' for 2d approach
                                      'popsize'= 50L,
                                      'termination'= py_none(), # By default it runs for 1000 generations (or pass a pymoo termination instance)
                                      'save_history'= FALSE, # set to True for later hypervolume calculations
                                      'seed'= py_none() # For reproducibility
                                    ))
)



expgen = moo_datagen$DataGenerator(expconfig=expconfig) # Pass defined parameters

datagen <- expgen$generate_data()

results <- moo_communities$run_serial_communities(datagen, algos)

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
