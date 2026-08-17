import igraph
import numpy as np
import code

class ExpConfig():
    '''
    This class defines the configuration parameters required
    to generate the dataset (graphs) for community detection 
    '''
    def __init__(
        self, L=[40,60], U=[40,60], NumEdges=200, BC=0.2,
        NumGraphs=30, shuffle=True, seed=None,filename=''):
        
        assert len(L) == len(U), 'This generator only supports the same number of communities in each mode'
        
        self.L = L # L vertices (lower graph part)
        self.U = U # U vertices (upper graph part)
        self.NumEdges=NumEdges
        
        self.BC = BC # Matching probabilities for the communities.
        self.NumGraphs = NumGraphs # Number of graphs to generate for this experiment

        self.NumNodes = sum(L)+sum(U) # Number of graph nodes
        self.Vertices = [0] * sum(self.L) + [1] * sum(self.U)
        self.shuffle=shuffle
        self.seed = 42 if seed is None else seed
        self.filename=filename
        
        ## Add a warning around the number of edges.
        max_in_comm_edges = sum([s*t for s,t in zip(L,U)])
        if NumEdges > max_in_comm_edges: print('The parameters specify more edges than the maximum within communities. Between community edge probability will be affected.')
    
    def __str__(self) -> str:
        return f'<ExpConfig: filename={self.filename}, L={self.L}, U={self.U}, NumNodes={self.NumNodes}, NumEdges={self.NumEdges}, BC={self.BC}, NumGraphs={self.NumGraphs}, shuffle={self.shuffle}, seed={self.seed}>'


class DataGenerator():
    def __init__(self, expconfig=None):
        self.expconfig = ExpConfig() if expconfig is None else expconfig
    
    def __str__(self) -> str:
        return f'<DataGenerator: {self.expconfig.__str__()[1:-1]}>'

    def generate_data(self):
        rng = np.random.default_rng(seed=self.expconfig.seed)
        shuffle = self.expconfig.shuffle
        
        L = self.expconfig.L
        U = self.expconfig.U
        BC = self.expconfig.BC
        filename = self.expconfig.filename
        vertices = self.expconfig.Vertices
        
        ## Make some useful calculations to save time.
        n_top = sum(U)
        n_bottom = sum(L)
        n_top_comms = len(U)
        n_bottom_comms = len(L)
        
        i = 0
        comm_labels = {}  ## This dict will give node id: community label with numbering sequentially across both modes, starting from 0.
        ## That is, the first node in the bottom set has id n_top and the first bottom community has id len(community_prefs[0]).
        for m in (L,U):
            for j,c in enumerate(m):
                for n in range(c):
                    comm_labels[i] = j
                    i += 1

        ## Reverse this dict to get a list of bottom node ids within each community label.
        comm_nodes = {}
        for k in comm_labels:
            if k >= n_bottom:  ## We only care about mapping the top nodes.
                if comm_nodes.get(comm_labels[k]):
                    comm_nodes[comm_labels[k]].append(k)
                else:
                    comm_nodes[comm_labels[k]] = [k]

        ## Make a dict of out of community maps to save time later.
        off_comms = {i:[j for j in range(n_top_comms) if j != i] for i in range(n_bottom_comms)}

        for it in range(self.expconfig.NumGraphs):
            ## Sample the bottom nodes for each edge.
            source_nodes = rng.choice(n_bottom,size=self.expconfig.NumEdges)

            ## Calculate the probability to determine the community of the bottom node for each edge.
            comm_probs = rng.random(self.expconfig.NumEdges)

            ## Turn the probabilities into communities to target.
            target_comms = [comm_labels[source_nodes[i]] if p > BC else rng.choice(off_comms[comm_labels[source_nodes[i]]]) for i,p in enumerate(comm_probs)]

            ## Sample the target nodes from the communities.
            target_nodes = [rng.choice(comm_nodes[c]) for c in target_comms]

            edges = list(zip(source_nodes,target_nodes))

            ## Resample any duplicate edges.
            while len(edges) != len(set(edges)):
                #print('resampling duplicate edges, %d to go' % (len(edges)-len(set(edges))))
                edges = list(set(edges))
                ## We have a duplicate edge, resample it.
                sn = rng.choice(n_bottom,size=self.expconfig.NumEdges-len(edges))
                cp = rng.random(self.expconfig.NumEdges-len(edges))
                tc = [comm_labels[sn[i]] if p > BC else rng.choice(off_comms[comm_labels[sn[i]]]) for i,p in enumerate(cp)]
                tn = [rng.choice(comm_nodes[c]) for c in tc]
                edges += list(zip(sn,tn))
     
        
            # Specify original ground truth
            groundtruth=[comm_labels[i] for i in range(sum(U)+sum(L))]
            # shapes = ["rectangle"] * int(L*ML) + ["circle"] * int(L*(1-ML)) + ["rectangle"] * int(U*MU) + ["circle"] * int(U*(1-MU))

            # Reduce to giant component
            g_i = igraph.Graph.Bipartite(vertices, edges)
        
            index_max = np.argmax(g_i.components().sizes()) # Get the largest graph component
            # print(g_i.components().sizes())
            
            if not shuffle:
                T = [groundtruth[i] for i in g_i.clusters()[index_max]]
                vT = [vertices[i] for i in g_i.clusters()[index_max]]
                g_new=g_i.clusters().giant()
                groundtruth=T
            
                # Create actual bipartite instance
                g_i_new = igraph.Graph.Bipartite(vT,g_new.get_edgelist())

                # Setting attributes
                g_i_new.vs['VX'] = vT # Vertices
                g_i_new.vs['name'] = vT # Vertices
                g_i_new.vs['GT'] = groundtruth # Ground truth
                
                ## Print gc stats for g_i_new.
                print('Graph giant component has %d/%d nodes and %d/%d edges and %d/%d communities.' % (len(g_i_new.vs),len(vertices),len(g_i_new.es),self.expconfig.NumEdges,len(set(g_i_new.vs['GT'])),len(self.expconfig.L)))

                ## Write g to file.
                if filename:
                    try:
                        g.write_gml(filename+'_%d.gml' % it)
                    except FileNotFoundError:
                        print('ERROR: You need to create the specified directory.')
                        raise FileNotFoundError
                yield g_i_new #, vT, groundtruth,
            else:
                #T = [groundtruth[i] for i in g_i.clusters()[index_max]]
                #vT = [vertices[i] for i in g_i.clusters()[index_max]]
                #g=g_i.clusters().giant()
                #print(g)
            
                oldelist = g_i.get_edgelist()
                oldorder= g_i.clusters()[index_max]#list(range(0,len(T)))
                order=oldorder.copy()
                #print(order)
                rng.shuffle(order)
                # random.shuffle(order)
                #print(groundtruth)
                #print(vT)
            #    print(oldelist)
                #
                #print(oldorder)
                #print(order)
                selected = g_i.clusters()[index_max]
                
                elist=[]
                
                # Create reduced edge list
                for i in range(0,len(oldelist)):
                    try :
                        # Retrieve old idnex
                        ti1 = oldorder.index(oldelist[i][0])
                        ti2 = oldorder.index(oldelist[i][1])
                        # Map vertices to new ids
                        i1 = order.index(oldelist[i][0])
                        i2 = order.index(oldelist[i][1])
                        elist.append([i1,i2])
                        #print(ti1,ti2)
                        #print(i1,i2)
                
                    except ValueError :
                        res = "Element not in list !"
                        # print(res)
                        

                    # print(elist[i])
                        #print(i1,i2)
                #    print(elist) 
                    
                # print(order)
                # print(len(elist))
                # print(len(oldelist))
                # print(T)
                # print(vT)

                labels = [groundtruth[i] for i in order]
                topbottom = [vertices[i] for i in order]
            #    print(topbottom)
                #print(labels)
                #print(topbottom)
                #print(groundtruth)
                #print(vT)
                #print(len(oldelist))
                #print(len(elist))
                
                # Create actual bipartite instance
                g = igraph.Graph.Bipartite(topbottom,elist)
                #g_i = Graph.Bipartite(vT,g.get_edgelist())

                # # Store instance and associated ground truth
                # g.write_edgelist(path+"Graph"+str(it)+".dat")
                # p = pd.DataFrame(labels)
                # p.to_csv(path+"Graph"+str(it)+".truth.dat", sep=',',header=None)
                # p = pd.DataFrame(topbottom)
                # p.to_csv(path+"Graph"+str(it)+".vertices.dat", sep=',',header=None)

                # Setting attributes
                g.vs['VX'] = topbottom # Vertices
                g.vs['name'] = topbottom # Vertices
                g.vs['GT'] = labels # Ground truth
                #code.interact(local=locals())

                ## Print gc stats for g_i_new.
                print('Graph giant component has %d/%d nodes and %d/%d edges and %d/%d communities.' % (len(g.vs),len(vertices),len(g.es),self.expconfig.NumEdges,len(set(g.vs['GT'])),len(self.expconfig.L)))

                ## Write g to file.
                if filename:
                    try:
                        g.write_gml(filename+'_%d.gml' % it)
                    except FileNotFoundError:
                        print('ERROR: You need to create the specified directory.')
                        exit()

                yield g#, vT, groundtruth,


def graphs_equal(g1, g2, attribs):
    """
    Checks the equality of two graphs
    Equal means vertex indices are equal, edge vertex indices are equal,
    and the provided vertex attributes are equal (if provided)
    """
    # Check indices
    if (g1.vs.indices != g2.vs.indices) or\
        (sorted(g1.get_edgelist()) != sorted(g2.get_edgelist())):
        return False
    
    # Check attributes
    for attr in attribs:
        if (g1.vs[attr] != g2.vs[attr]):
            return False

    return True


def test_create_datagen():
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=1000000, size=1)
    # expconfig= ExpConfig(L=[8,12], U=[40,60], NumEdges=200, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
    expconfig= ExpConfig(L=[40,60], U=[200,300], NumEdges=1000, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)

    print(expconfig)
    expgen = DataGenerator(expconfig=expconfig)
    print(expgen)
    datagen = expgen.generate_data()
    for i, it in enumerate(datagen, 1):
        # graph, vertices, groundtruth = it
        graph = it
        print(f"generated graph {i} : V {len(graph.vs)}, E:{len(graph.es)}")
        # break


def test_datagen_reproducibility(num_tests=1, attribs=['VX', 'GT']):
    print(f"Testing {num_tests} data generation configuration(s)")
    print(f"Using attribs {attribs}")
    for i in range(num_tests):
        different=False
        rng = np.random.default_rng()
        seed = rng.integers(low=0, high=1000000, size=1)
        print(f"\tConfig {i} using seed {seed}:", sep= ' ')
        # expconfig1= ExpConfig(L=[8,12], U=[40,60], NumEdges=200, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
        # expconfig2= ExpConfig(L=[8,12], U=[40,60], NumEdges=200, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
        expconfig1= ExpConfig(L=[40,60], U=[200,300], NumEdges=1000, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)
        expconfig2= ExpConfig(L=[40,60], U=[200,300], NumEdges=1000, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)
        expgen1 = DataGenerator(expconfig=expconfig1)
        expgen2 = DataGenerator(expconfig=expconfig2)
        datagen1 = expgen1.generate_data()
        datagen2 = expgen2.generate_data()
        for idx, (g1, g2) in enumerate(zip(datagen1, datagen2)):
            if not graphs_equal(g1, g2, attribs):
                different = True
                break
        if different:
            print(f"\t\tGraphs at index {idx} are not equal")
            break
        else:
            print(f"\t\tAll Graphs are equal")

if __name__ == "__main__":
    # test_create_datagen()
    test_datagen_reproducibility(num_tests=30, attribs=['VX', 'GT'])
