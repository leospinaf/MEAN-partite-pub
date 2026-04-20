## This script loads the crossover functions implemeted and tests them on a given example.

from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
import code
import numpy as np


## Define a set of candidates.
#X = np.array([[[i]*10 for i in range(2)] for _ in range(3)])  ## This is the wrong shape, the first dimension should be 2.
#X = np.array([[[0,1] for _ in range(3)] for _ in range(10)])  ## Again the wrong shape. This is shape 10x3x2
X = np.array([[[0]*10 for _ in range(10)],[[1]*10 for _ in range(10)]])

for i in np.arange(0,1,0.1):
    crossover=get_crossover("int_ux", prob=i) # HParams to test
    out = crossover._do(0,X)
    print(i)
    print(out.sum(axis=2)/10)
    print(out)

class dummy_problem():
	def __init__(self):
		self.n_var = 10
dp = dummy_problem()
## Repeat using .do instead. Currently does not work given the need to pass carefully constructed parents (expects array of pair ids) and population (all candidates).
#for i in np.arange(0,1,0.1):
#    crossover=get_crossover("int_ux", prob=i) # HParams to test
#    code.interact(local=locals())
#    out = crossover.do(dp,X,X)
#    print(i)
#    print(out.sum(axis=2)/10)
#    print(out)

from pymoo.core.crossover import Crossover
def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]
    return _X

class CustomCrossover(Crossover):

    def __init__(self, p_cross_pair=1,p_cross_gene=0.1, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.p_cross_pair = p_cross_pair ## This should be handled by kwarg prob
        self.p_cross_gene = p_cross_gene

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < self.p_cross_gene
        ## Now clear the rows that shouldn't swapped.
        for i,p in enumerate(np.random.random(n_matings)):
            if p > self.p_cross_pair:
                M[i,:] = False
        _X = crossover_mask(X, M)
        return _X

X = np.array([[[0]*10 for _ in range(10)],[[1]*10 for _ in range(10)]])
for j in np.arange(0,1,0.1):
    custom_crossover = CustomCrossover(p_cross_gene=j,prob=0.1)
    out_cust = custom_crossover._do(0,X)
    print(j)
    print(out_cust.sum(axis=2)/10)
    print(out_cust)

code.interact(local=locals())