## This script tests individual features added in the newest version for correctness.

import moo.data_generation as data_generation
import moo.contestant as contestant
import moo.multicriteria as multicriteria
import code
import sknetwork

## Check seed is set properly in the generator, i.e. two instances with the same parameters give the same net.
expconfig = data_generation.ExpConfig(
	L=[15,15], U=[15,15], NumEdges=200, BC=0.1, NumGraphs=1,
	shuffle=False, seed=24
	)

expgen = data_generation.DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()
g = next(datagen)
datagen = expgen.generate_data()
h = next(datagen)

print('Regeneration of data respects the same seed:')
print(g.isomorphic(h))

## Check the calculation of Barber's modularity.
edge_list = [(0,1),(0,2),(1,0),(1,1),(1,4),(2,1),(3,2),(3,3),(3,4),(4,0),(4,3)]
low_gt = [0,0,0,1,1]
up_gt = [0,0,0,1,1]
badj = sknetwork.utils.edgelist2biadjacency(edge_list)

qb = sknetwork.clustering.bimodularity(badj,low_gt,up_gt)
true_qb = 26/121
print('Error in Barber modularity calculation: %f' % (abs(qb-true_qb)))

## Check the calculation of Murata's modularity.
qm = contestant.modularity_murata(badj,low_gt+up_gt)
true_qm = 115/484
print('Error in Murata modularity calculation: %f' % (abs(qm-true_qm)))

code.interact(local=locals())