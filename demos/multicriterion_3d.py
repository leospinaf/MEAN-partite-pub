# path = "/Users/mcdicjh2/Desktop/gecco22/"
# path = "./_temp/data_2_shuffled_fig06/"
# path = "./_temp/data_1_nonshuffled/" # for non-dominated solution issue
path = './_temp/fig07_data_1_nonshuffled/'
print(path)
from sklearn.metrics.cluster import adjusted_rand_score

# Recursive function to support translation of MST into initial GA solutions

def recursive_links(node_index, node_origin, x, temp_edges, binary_links, a, mst):
    

   # print(node_index,node_origin)
    
    # Draw link 
    if x[node_index] != -1:
        #print("I have already been here - going backwards")
        return

    # Set link to originating node - # Decode corresponding to full adjacency matrix
    for j in range(0,len(a[node_index])):        
        if a[node_index][j] == node_origin:
            x[node_index] = j+1
            binary_links[node_index] = node_origin
            temp_edges.append([node_index,node_origin]) # translation from node indices to indices in the adj list
                
    # Start recursion on all neighbors in MST
    for i in range(0,len(mst[node_index])):
        recursive_links(mst[node_index][i],node_index, x, temp_edges, binary_links, a, mst)
    
    return


# Set up EA in pymoo (3D)
import numpy as np
import pandas as pd
import random
from pymoo.util.misc import stack
# from pymoo.model.problem import Problem
from pymoo.core.problem import Problem, ElementwiseProblem
import igraph
from igraph import *

# from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination

termination = get_termination("n_gen", 1000) 
popsize = 50  


# PyMOO Problem Definition for the GA
# class MyProblem(Problem):
class MyProblem3D(ElementwiseProblem):
    def __init__(self,g1,g2,g3,a,vertices,xl,d,binary_links,n_var,lower):
        self.g1=g1
        self.g2=g2
        self.g3=g3
        self.a=a # adj mx
        self.vertices=vertices
        self.xl=xl # lower/upper boundaries (d-ring neighbors in tha adj mx)
        self.xu=d
        self.binary_links=binary_links # may not be used?
        self.n_var=n_var # redundant
        self.lower=lower # 
        super().__init__(n_var=n_var,
                         n_obj=3, # update to 2D
                         n_constr=0,
                         # Set upper bound for each decision variable using information about node degrees
                         xl=xl,                         
                         xu=d,
                        #  elementwise_evaluation=True
                         )


    def _evaluate(self, x, out, *args, **kwargs): # Mutivar fitness function
        f1 = 0
        f2 = 0
        f3 = 0
        ctr0 = 0
        ctr1 = 0
        
        sol_edges = []
        # Allow self loops in encoding which are interpreted as "no edges"
        for i in range(0,self.n_var):
            if x[i] > 0:
                sol_edges.append([i,self.a[i][x[i]-1]]) # x sol we are looking for

         
        # Construct bipartite graph
        t = Graph.Bipartite(self.vertices,sol_edges)
        
        # Decoding: Identify unconnected components to define communities
        c = t.clusters()
        m = c.membership
        k = max(c.membership)+1
        
        proj0 = [i for i,val in enumerate(self.vertices) if val==0]
        proj1 = [i for i,val in enumerate(self.vertices) if val==1]
        
        proj0_labels=[m[i] for i in proj0]
        proj1_labels=[m[i] for i in proj1]
        
        # Evaluate both projections with respect to those communities
        m1 = self.g1.modularity(proj0_labels,weights=self.g1.es['weight'])#
        m2 = self.g2.modularity(proj1_labels,weights=self.g2.es['weight'])#
        m3 = self.g3.modularity(m)
        
        
        out["F"] = [-m1, -m2, k] # k number of connected components (3D), 2D [-m3, k]


# Now actually run the GA
from pymoo.optimize import minimize
import multiprocessing
import time

temp_results = [] # Results for all graphs, after duplicated ( per graph are removed)

    
def worker3D(name):

    temp_results2 = [] # Results for one graph (no duplicates)
    # print("Starting")
    # name = multiprocessing.current_process().name
    print(f"Processing graph {int(name)+1}")
 
    # Read in graph and associated data
    g_org = Graph.Read_Edgelist(path+"Graph"+name+".dat")
    truth = pd.read_csv(path+"Graph"+name+".truth.dat", sep=',',header=None)
    groundtruth=truth.iloc[:, 1].values.tolist()
    v = pd.read_csv(path+"Graph"+name+".vertices.dat", sep=',',header=None)
    vertices=v.iloc[:, 1].values.tolist()
    edges = g_org.get_edgelist()
    
    proj0 = [i for i,val in enumerate(vertices) if val==0]
    proj1 = [i for i,val in enumerate(vertices) if val==1]
 
    # Construct bipartite instance
    g3 = Graph.Bipartite(vertices, edges)
    n_var = len(vertices)
    lower = vertices.count(0)
    

    # Determine both projections
    g1, g2 = g3.bipartite_projection(multiplicity=True)

    # Determine adjacency list
    a = g3.get_adjlist()


    # Determine number of possible edges for mutation (upper bound)
    d = g3.indegree()

    xl = np.empty(n_var) # lower bound
    xl.fill(0)

    

    
    
    # Generate Minimum Spanning Tree

    t = g3.spanning_tree(weights=g3.edge_betweenness()) # initial solution for the GA (specialize the resulting spanning tree)
    # print(t)

    e = t.get_edgelist()

    adj = t.get_adjlist()

    x = np.empty(n_var)
    x.fill(-1)
    x = x.tolist()

    binary_links = np.empty(n_var)
    binary_links.fill(-1)
    binary_links = binary_links.tolist() # May be got rid of?
    
    
    elementwise_problem = MyProblem3D(g1,g2,g3,a,vertices,xl,d,binary_links,n_var,lower)
    mp_result = pd.DataFrame(columns=['k','m', 'm1', 'm2', 'ari'])

    for it in range (0,1): # Unnecessary

       # Minimum Spanning Tree Initialization        (2 steps)
        mst = t.get_adjlist()

        temp_edges = []
        for i in range(0,n_var):
            if x[i] == -1:
                x[i] = 0
                for j in range(0,len(mst[i])):
                    recursive_links(mst[i][j],i, x, temp_edges, binary_links, a, mst) # translate MST into a genome
           
            
        pop = np.tile(x, (popsize, 1)) # Generate the initial generation (identical genomes/solutions)
        c= g3.clusters()
        ctr = 0 

        # Initialization: Fast Greedy (diversity in the initial generation)
        for i in range(len(c),1+min(50,popsize,n_var)):

            # Use different greedy solutions for diversity
            test = g3.community_fastgreedy()
            test = test.as_clustering(i)#i

            test=test.membership
            for j in range(0,n_var):
                if pop[ctr][j] != 0 and test[a[j][pop[ctr][j]-1]] != test[j]: # Removing edges crossing communities
                    pop[ctr][j] = 0

            ctr = ctr +1

        # Initialization: Random Solutions (fill up the generation with additional solutions) check if used below
        for i in range(ctr,popsize):
            maxk = random.randint(0,20)
            for k in range(0,maxk):
                index = random.randint(0,n_var-1)
                pop[i][index] = 0


        algorithm = NSGA2(
            pop_size=popsize,
            n_offsprings=popsize,
            sampling=pop,
            crossover=get_crossover("int_ux", prob=0.3), # HParams to test
            mutation=get_mutation("int_pm"), # HParams to test
            eliminate_duplicates=True
        )
        # Popsize, number of generations more important that the above HParams
        # Check hypervolume for convergence test to avoid long running times (early stopping)



        res = minimize(elementwise_problem,
                   algorithm,
                   termination,
                   seed=None,
                   save_history=True,
                   verbose=False, # True
                   )

        # Collate results and eliminate duplicates
        # print(range(0,len(res.X)))

        for n in range(0,len(res.X)):
            sol_edges = []
            for i in range(0,n_var):
                if res.X[n][i] > 0:
                    sol_edges.append([i,a[i][res.X[n][i]-1]])
            t = Graph.Bipartite(vertices,sol_edges)
            c= t.clusters()

            m = c.membership
            m3 = g3.modularity(m)
            ari = adjusted_rand_score(groundtruth,m)
            
            proj0_labels=[m[i] for i in proj0]
            proj1_labels=[m[i] for i in proj1]
            
            m1 = g1.modularity(proj0_labels,weights=g1.es['weight'])#
            m2 = g2.modularity(proj1_labels,weights=g2.es['weight'])


            mp_result.loc[it*30+n] = len(c), m3, m1, m2, ari
            temp_results2.append(['3d', len(c), m3, m1, m2, ari, int(name)+1])

    import pickle
    # pickle.dump(mp_result, open('./_temp/nondominant_duplicates.pickle', 'wb'))
    mp_result.drop_duplicates(keep="first",inplace=True)
    # pickle.dump(mp_result, open('./_temp/nondominant.pickle', 'wb'))
    mp_result.to_csv(path+"Graph"+name+".mp.csv")
    print(path+"Graph"+name+".mp.csv")

    cols = ['name', 'num_clusters', 'modularity_score', 'modularity_score_1', 'modularity_score_2', 'adj_rand_index', 'graph_idx']
    df_3d_legacydata_legacycode = pd.DataFrame(columns=cols, data=temp_results2)
    # print("Before duplicate removal, size :", df_3d_legacydata_legacycode.shape)
    df_3d_legacydata_legacycode.drop_duplicates(keep="first",inplace=True)
    # print("After duplicate removal, size :", df_3d_legacydata_legacycode.shape)
    temp_results2 = df_3d_legacydata_legacycode.values.tolist() #TODO is tolist necessary?
    temp_results.extend(temp_results2)
    print(f"Done Processing graph {int(name)+1}")

# Parallelization over potentially multiple runs (change 1 below e.g. to 30)
# for run in range(0,30):
#     print(run)
#     worker1 = multiprocessing.Process(name=str(run), target=worker)
#     worker1.start()


if __name__ == "__main__":
    from time import perf_counter
    start= perf_counter()

    # # Parallelization over potentially multiple runs (change 1 below e.g. to 30)
    # for run in range(0,30):
    #     print(run)
    #     worker1 = multiprocessing.Process(name=str(run), target=worker3D)
    #     worker1.start()

    # # For non-dominant solution issue
    # worker3D(str(20))

    # Serial
    for run in range(0,30):
        # worker1 = multiprocessing.Process(name=str(run), target=worker)
        # worker1.start()
        worker3D(str(run))
        print(run)
        # break
        # if run == 2:
        #     break

    end = perf_counter()
    print("elapsed time ", end -start)

    cols = ['name', 'num_clusters', 'modularity_score', 'modularity_score_1', 'modularity_score_2', 'adj_rand_index', 'graph_idx']
    df_3d_legacydata_legacycode = pd.DataFrame(columns=cols, data=temp_results)
    path = './_temp'
    os.makedirs(path, exist_ok=True)
    df_3d_legacydata_legacycode.to_csv(os.path.join(path, 'fig_07_nopert_3d.csv'), index=None)
    df_3d_legacydata_legacycode.shape


