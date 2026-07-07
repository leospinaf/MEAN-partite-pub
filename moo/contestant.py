import numpy as np
import pandas as pd
import igraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import condor
from moo.utils import nostdout
import sknetwork
import cdlib
import moo.skbio_gini as skbio_gini

class CommunityDetector():
    """
    Base class for community detection
    Inspired from the Estimator API of scikit-learn, cf. https://scikit-learn.org/stable/developers/develop.html
    Results are in self.params_ dictionary
    """

    def __init__(self, name="") -> None:
        # Any parameters not related to the data (Graph)
        # need to be defined here and have default values (in subclasses)
        self.name_ = name
        self.params_ = dict() # Parameters of the community detector
        self.results_ = [] # Results (list of dictionaries)
        
    def check_graph(self, graph):
        assert isinstance(graph, igraph.Graph), "graph must be of type igraph.Graph"
        assert graph.is_bipartite(return_types=False), "graph must be a bipartite graph"
        assert graph.is_connected(), "graph must be fully connected (one connected component)"
        assert len(graph.vs), "graph must not be empty"
        
    def compute_communities(self, graph, y=None): # graph is a bipartite graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        return self # Needs to return self

    def get_results(self):
        # Returns the community detection results
        return self.results_
    
    def get_params(self):
        # Returns the community detection parameters
        return self.params_


class ComDetFastGreedy(CommunityDetector):
    def __init__(self, name= "fastgreedy", params = {'weights': None}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_fastgreedy(**self.params_)
       
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        badj = make_badj(self.graph_)
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_barber = sknetwork.clustering.bimodularity(badj,proj0_labels,proj1_labels)
            modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            
            ## Add in the other metrics. First need to create a cdlib clustering.
            communities = [[] for i in range(max(vx_clustering.membership)+1)] ## List of list of node ids.
            for i,lab in enumerate(vx_clustering.membership):
                communities[lab].append(i)
            clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
            conductance = cdlib.evaluation.conductance(self.graph_,clust).score
            coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
            performance = bi_performance(badj, proj0_labels + proj1_labels)
            gini = skbio_gini.gini_index([len(c) for c in communities])
            
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_barber = modularity_score_barber,
                modularity_score_murata = modularity_score_murata,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
                conductance = conductance,
                coverage = coverage,
                performance = performance,
                gini = gini
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetEdgeBetweenness(CommunityDetector):
    def __init__(self, name= "edgebetweenness", params = {'directed': False, 'weights': None}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_edge_betweenness(**self.params_)

        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        badj = make_badj(self.graph_)
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_barber = sknetwork.clustering.bimodularity(badj,proj0_labels,proj1_labels)
            modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            
            communities = [[] for i in range(max(vx_clustering.membership)+1)] ## List of list of node ids.
            for i,lab in enumerate(vx_clustering.membership):
                communities[lab].append(i)
            clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
            conductance = cdlib.evaluation.conductance(self.graph_,clust).score
            coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
            performance = bi_performance(badj, proj0_labels+proj1_labels)
            gini = skbio_gini.gini_index([len(c) for c in communities])
            
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_barber = modularity_score_barber,
                modularity_score_murata = modularity_score_murata,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
                conductance = conductance,
                coverage = coverage,
                performance = performance,
                gini = gini
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetWalkTrap(CommunityDetector):
    def __init__(self, name= "walktrap", params = {'weights': None, 'steps': 4}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_walktrap(**self.params_) #? steps changed

        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        badj = make_badj(self.graph_)
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_barber = sknetwork.clustering.bimodularity(badj,proj0_labels,proj1_labels)
            modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            
            communities = [[] for i in range(max(vx_clustering.membership)+1)] ## List of list of node ids.
            for i,lab in enumerate(vx_clustering.membership):
                communities[lab].append(i)
            clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
            conductance = cdlib.evaluation.conductance(self.graph_,clust).score
            coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
            performance = bi_performance(badj, proj0_labels+proj1_labels)
            gini = skbio_gini.gini_index([len(c) for c in communities])
            
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_barber = modularity_score_barber,
                modularity_score_murata = modularity_score_murata,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
                conductance = conductance,
                coverage = coverage,
                performance = performance,
                gini = gini
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetMultiLevel(CommunityDetector):
    def __init__(self, name= "multilevel", params = {'weights': None, 'return_levels': False}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        # Run Multi-Level algorithm (not implemented in igraph package)
        res1 = graph_proj1.community_multilevel(**self.params_)
        res2 = graph_proj2.community_multilevel(**self.params_)

        # # Cluster assignment from each projection
        # assignment=res1.membership + res2.membership

        # Consider perturbation in creating new membership vector
        it1=0
        it2=0
        assignment = [0] * len(vertices)
        for vit in range(0, len(vertices)):
            if vertices[vit] == 0:
                assignment[vit] = res1.membership[it1]
                it1=it1+1
            else:
                assignment[vit] = res2.membership[it2]
                it2=it2+1

        k1 = max(res1.membership) + 1
        k2 = max(res2.membership) + 1
        d = np.zeros(shape=(k1+k2, k1+k2))

        # Calculate dissimilarity matrix between communities (rows/columns are community indices in the 2 projected graph, and values are the number of edges linking those communities (vertices))
        for ei in range(0, len(edges)): # For each edge in the bipartite graph
            #print(edges[ei],edges[ei][0],edges[ei][1])
            index1 = assignment[edges[ei][0]] # Get community index of the edge's source vertex in the first one-mode projection
            index2 = k1 + assignment[edges[ei][1]] # Get community index of the edge's target vertex in the second one-mode projection # Why assigning different communitites to one vertex? A guarantee that the 2nd vertex of each edge belongs to the second one-mode projection?
            if vertices[edges[ei][0]] == 0: # a (0,1) edge
                index1=assignment[edges[ei][0]]
                index2=k1+assignment[edges[ei][1]]
            else: # a (1,0) edge
                index1=k1+assignment[edges[ei][0]]
                index2=assignment[edges[ei][1]]

            # print(index1,index2)
            d[index1][index2] += 1 # Update matrix item
            d[index2][index1] += 1 # Update matrix item
        
        # Normalize (adding 1 to avoid division by zero) and setting the matrix main diagonal to zero
        for d1 in range(0, k1+k2):
            for d2 in range(0, k1+k2):
                d[d1][d2] = 1.0/(1.0+d[d1][d2])
            d[d1][d1] = 0
        
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        for k in range(1, k1+k2):
            # Run hierarchical clustering on communities
            clustering = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='precomputed').fit(d)
            labels = clustering.labels_

            newlabels = np.zeros(n_vertices)

            # for v in range(0,lower):
            #     newlabels[v] = labels[assignment[v]]

            # for v in range(lower,n_vertices):
            #     newlabels[v] = labels[k1 + assignment[v]]

            for v in range(0,n_vertices):
                if vertices[v] == 0:
                    newlabels[v] = labels[assignment[v]]
                else:
                    newlabels[v] = labels[k1+assignment[v]]

            proj0_labels=[newlabels[i] for i in proj0]
            proj1_labels=[newlabels[i] for i in proj1]
            badj = make_badj(self.graph_)
            modularity_score = self.graph_.modularity(newlabels)
            
            modularity_score_barber = sknetwork.clustering.bimodularity(badj,list(map(int,proj0_labels)),list(map(int,proj1_labels)))
            modularity_score_murata = modularity_murata(badj,list(map(int,proj0_labels+proj1_labels)))
            # modularity_score_1 = graph_proj1.modularity(newlabels[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(newlabels[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, newlabels)
            
            communities = [[] for i in range(int(max(newlabels))+1)] ## List of list of node ids.
            for i,lab in enumerate(newlabels):
                communities[int(lab)].append(i)
            clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
            conductance = cdlib.evaluation.conductance(self.graph_,clust).score
            coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
            performance = bi_performance(badj, proj0_labels+proj1_labels)
            gini = skbio_gini.gini_index([len(c) for c in communities])
            
            result = dict(
                    name=self.name_,
                    num_clusters = k,
                    modularity_score = modularity_score,
                    modularity_score_barber = modularity_score_barber,
                    modularity_score_murata = modularity_score_murata,
                    modularity_score_1 = modularity_score_1,
                    modularity_score_2 = modularity_score_2,
                    adj_rand_index = adj_rand_index,
                    conductance = conductance,
                    coverage = coverage,
                    performance = performance,
                    gini = gini
                )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetBRIMNoPert(CommunityDetector):
    def __init__(self, name= "brim", params = {'method': 'LCS', 'project': False}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        ground_truth = self.graph_.vs['GT']
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        net = pd.DataFrame(edges, dtype=str)

        with nostdout():
            co = condor.condor_object(net)
            co = condor.initial_community(co, **self.params_)
            #co['reg_memb']['community'] = (co['reg_memb']['community'] % 2)
            
        if max(co['reg_memb'].iloc[:,1])+1 > self.max_num_clusters_:
            print('BRIM found too many communities in the initial assignment. Try increasing max_num_clusters above %d.' % self.max_num_clusters_)
            exit()
        with nostdout():
            co = condor.brim(co,c=self.max_num_clusters_)#,c=max(co['reg_memb'].iloc[:,1])+1)

        groundtruth1 = ground_truth[0:lower]
        groundtruth2 = ground_truth[lower:n_vertices]

        # output1 = co["reg_memb"]
        # output1 = output1["com"].tolist()
        # output2 = co["tar_memb"]
        # output2 = output2["com"].tolist()

        output1=co['reg_memb']
        # print(output1.shape)
        output1=output1["com"].tolist()
        output2 = co['tar_memb']
        # print(output2.shape)
        output2=output2["com"].tolist()
        
        # adj_rand_index_1 = adjusted_rand_score(groundtruth1, output2)
        # adj_rand_index_2 = adjusted_rand_score(groundtruth2, output1)
        output3 = output2 + output1
        adj_rand_index = adjusted_rand_score(ground_truth, output3)
        badj = make_badj(self.graph_)
        modularity_score = self.graph_.modularity(output3)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj,output1,output2)
        modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
        # print(len(output2), len(graph_proj1.vs), len(graph_proj1.es['weight']))
        # print(len(output1), len(graph_proj2.vs), len(graph_proj2.es['weight']))
        modularity_score_1 = graph_proj1.modularity(output1, weights = graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(output2, weights = graph_proj2.es['weight'])

        k = (max(output3) + 1)

        communities = [[] for i in range(max(output3)+1)] ## List of list of node ids.
        for i,lab in enumerate(output3):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_,clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
        performance = bi_performance(badj, output3)
        gini = skbio_gini.gini_index([len(c) for c in communities])
            
        result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_barber = modularity_score_barber,
                modularity_score_murata = modularity_score_murata,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
                conductance = conductance,
                coverage = coverage,
                performance = performance,
                gini = gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetBRIM(CommunityDetector):
    def __init__(self, name= "brim", params = {'method': 'LCS', 'project': False}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        #FIXME - min_num_clusters and max_num_clusters not making it to co object
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

        self.__test_condor_version()


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __test_condor_version(self):
        # Check we're using the old condor version.  Do this by trying to initialise a condor object with a dataframe 
        # parameter. This will only succeed on the new version.
        
        # Dummy dataset
        df = pd.DataFrame(list(zip(["0","2"], ["1","3"])),
               columns =['0', '1']) 
        df["weight"]=1

        try: 
            tc = condor.condor_object(dataframe=df)
        except TypeError as e:
            return True

        raise RuntimeError("Incorrect version of condor installed - use git commit 38993 from /genisott/pycondor")            
        
            
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        ground_truth = self.graph_.vs['GT']
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        # Fix edgelist representation for BRIM package (needs 0 vertices as start)
        for i in range (0, len(edges)):
            if vertices[edges[i][0]] == 0:
                temp = edges[i]
                edges[i] = (temp[0], temp[1])
            elif vertices[edges[i][1]] == 0:
                temp = edges[i]
                edges[i] = (temp[1], temp[0])
            else:
                print("Error")

        #code.interact(local=locals())
        net = pd.DataFrame(edges, dtype=str)

        # Set weight to 1 for all links
        # TODO add note
        net["weight"]=1

        # Run the algorithm, suppressing its very verbose output
        #code.interact(local=locals())
        with nostdout():
            co = condor.condor_object(net)
            co = condor.initial_community(co, **self.params_)
            #co['reg_memb']['community'] = (co['reg_memb']['community'] % 2)
            
        if max(co['reg_memb'].iloc[:,1])+1 > self.max_num_clusters_:
            raise ValueError('BRIM found too many communities in the initial assignment. Try increasing max_num_clusters above %d.' % self.max_num_clusters_)
        with nostdout():
            co = condor.brim(co,c=self.max_num_clusters_)#,c=max(co['reg_memb'].iloc[:,1])+1)

        # groundtruth1 = ground_truth[0:lower]
        # groundtruth2 = ground_truth[lower:n_vertices]

        # output1 = co["reg_memb"]
        # output1 = output1["com"].tolist()
        # output2 = co["tar_memb"]
        # output2 = output2["com"].tolist()

        # Get the original node numbers from the graph we gave condor
        #reg_memb = co.reg_memb.copy()
        reg_memb = co["reg_memb"].copy()
        reg_memb["reg"]=reg_memb["reg"].str.replace(r'^reg_', '', regex=True)
        reg_memb["reg"]=reg_memb["reg"].astype(int)
        reg_memb.rename(columns={"reg": "vindex"},inplace=True)
        reg_memb.sort_values("vindex", inplace=True)

        
        # tar_memb = co.tar_memb.copy()
        tar_memb = co["tar_memb"].copy()
        tar_memb["tar"]=tar_memb["tar"].str.replace(r'^tar_', '', regex=True)
        tar_memb["tar"]=tar_memb["tar"].astype(int)
        tar_memb.rename(columns={"tar": "vindex"},inplace=True)
        tar_memb.sort_values("vindex", inplace=True)

        combined_memb = pd.concat([reg_memb, tar_memb])
        combined_memb.sort_values("vindex", inplace=True)

        # Concatenate the two membership lists and sort by 

        # output1=co.reg_memb
        # output1=output1["com"].tolist()
        # output2 = co.tar_memb
        # output2=output2["com"].tolist()
        
        # adj_rand_index_1 = adjusted_rand_score(groundtruth1, output2)
        # adj_rand_index_2 = adjusted_rand_score(groundtruth2, output1)
        # output3 = output2 + output1

        

        # output3 = np.full(n_vertices, -1, dtype=int)
        # index1 = 0
        # index2 = 0
        # for v in range (0, n_vertices):
        #     if vertices[v] == 0:
        #         output3[v] = output2[index1]
        #         index1 += 1
        #     else:
        #         output3[v] = output1[index2]
        #         index2 += 1

        adj_rand_index = adjusted_rand_score(ground_truth, combined_memb["com"].tolist())
        badj = make_badj(self.graph_)
        modularity_score = self.graph_.modularity(combined_memb["com"].tolist())
        modularity_score_barber = sknetwork.clustering.bimodularity(badj,tar_memb["com"].tolist(),reg_memb["com"].tolist())
        modularity_score_murata = modularity_murata(badj,combined_memb["com"].tolist())
        modularity_score_1 = graph_proj1.modularity(tar_memb["com"].tolist(), weights = graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(reg_memb["com"].tolist(), weights = graph_proj2.es['weight'])

        k = (max(combined_memb["com"].tolist()) + 1)
        communities = [[] for i in range(max(combined_memb["com"].tolist())+1)] ## List of list of node ids.
        for i,lab in enumerate(combined_memb["com"].tolist()):
            communities[lab].append(i)
        communities = [c for c in communities if c]  ## Drop any empty communities from missing numbers that confuse cdlib
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)

        conductance = cdlib.evaluation.conductance(self.graph_,clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
        performance = bi_performance(badj, tar_memb["com"].tolist()+reg_memb["com"].tolist())
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_barber = modularity_score_barber,
                modularity_score_murata = modularity_score_murata,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
                conductance = conductance,
                coverage = coverage,
                performance = performance,
                gini = gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

#TODO ComDetBRIM (perturbation version) based on ComDetBRIMNoPert
#TODO Compare condor results for the current version and the old version 1.1 that Julia used

class ComDetBiLouvain(CommunityDetector):
    def __init__(self, name= "bilouvain", params = {'weights': None}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        
        badj = make_badj(self.graph_)
        ## Set up the biLouvain method. sknetwork rolls them both into one.
        bilouvain = sknetwork.clustering.Louvain()
        
        ## Now we fit bilouvain to the graph.
        bilouvain.fit(badj,force_bipartite=True)
        #code.interact(local=locals())
        proj0_labels=list(bilouvain.labels_row_)
        proj1_labels=list(bilouvain.labels_col_)
        graph_labels = [0]*len(ground_truth)
        for i,lab in zip(proj0,proj0_labels):
            graph_labels[i] = lab
        for i,lab in zip(proj1,proj1_labels):
            graph_labels[i] = lab
        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj,proj0_labels,proj1_labels)
        
        modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1,proj0_labels+proj1_labels)
        
        communities = [[] for i in range(max(proj0_labels+proj1_labels)+1)] ## List of list of node ids.
        for i,lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_,clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])
        
        result = dict(
            name=self.name_,
            num_clusters = (max(proj0_labels+proj1_labels)+1),
            modularity_score = modularity_score,
            modularity_score_barber = modularity_score_barber,
            modularity_score_murata = modularity_score_murata,
            modularity_score_1 = modularity_score_1,
            modularity_score_2 = modularity_score_2,
            adj_rand_index = adj_rand_index,
            conductance = conductance,
            coverage = coverage,
            performance = performance,
            gini = gini
            )
        self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

import pymocd

class ComDetariadne(CommunityDetector):
    def __init__(self, name="ariadne", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run Ariadne directly on the whole bipartite igraph object. pymocd's
        ## igraph ingestion uses vertex.index as the node id, which matches
        ## self.graph_.vs ordering exactly, so we can read the result straight
        ## back into a graph_labels list.
        raw_partition = pymocd.ariadne(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Ariadne assigns isolated nodes the shared sentinel community -1,
        ## but they aren't actually connected to one another, so give each
        ## isolated node its own singleton community instead of lumping them
        ## together. The rest of this pipeline (igraph.modularity, list-
        ## indexed community buckets) expects contiguous non-negative labels
        ## starting at 0, so remap real communities first, then append one
        ## fresh id per isolated node.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDethpmocd(CommunityDetector):
    def __init__(self, name="hpmocd", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run hpmocd
        raw_partition = pymocd.hpmocd(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

class ComDetmocd_q(CommunityDetector):
    def __init__(self, name="mocd_q", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run mocd_q
        raw_partition = pymocd.mocd_q(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

class ComDetmocd_d(CommunityDetector):
    def __init__(self, name="mocd_d", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run mocd_d
        raw_partition = pymocd.mocd_d(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetmoga_net(CommunityDetector):
    def __init__(self, name="moga_net", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run moga_net
        raw_partition = pymocd.moga_net(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

class ComDetccm(CommunityDetector):
    def __init__(self, name="ccm", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run ccm
        raw_partition = pymocd.ccm(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

class ComDetkrm(CommunityDetector):
    def __init__(self, name="krm", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run krm
        raw_partition = pymocd.krm(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

class ComDetmmcomo(CommunityDetector):
    def __init__(self, name="mmcomo", params={}, min_num_clusters=1, max_num_clusters=30) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params

        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        ## To work with the bilabels, we need to make gt0 and gt1 list proj0.
        gt0 = [v['GT'] for v in self.graph_.vs if v['type'] == 0]
        gt1 = [v['GT'] for v in self.graph_.vs if v['type'] == 1]
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        badj = make_badj(self.graph_)

        ## Run mmcomo
        raw_partition = pymocd.mmcomo(self.graph_)
        raw_graph_labels = [raw_partition[i] for i in range(n_vertices)]

        ## Make sure singletons are labelled as their own communities.
        real_labels = sorted(set(raw_graph_labels) - {-1})
        relabel = {lab: idx for idx, lab in enumerate(real_labels)}
        next_id = len(real_labels)
        graph_labels = []
        for lab in raw_graph_labels:
            if lab == -1:
                graph_labels.append(next_id)
                next_id += 1
            else:
                graph_labels.append(relabel[lab])

        proj0_labels = [graph_labels[i] for i in proj0]
        proj1_labels = [graph_labels[i] for i in proj1]

        modularity_score = self.graph_.modularity(graph_labels)
        modularity_score_barber = sknetwork.clustering.bimodularity(badj, proj0_labels, proj1_labels)

        modularity_score_murata = modularity_murata(badj, proj0_labels+proj1_labels)
        modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
        adj_rand_index = adjusted_rand_score(gt0+gt1, proj0_labels+proj1_labels)

        communities = [[] for i in range(max(graph_labels)+1)] ## List of list of node ids.
        for i, lab in enumerate(graph_labels):
            communities[lab].append(i)
        clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
        conductance = cdlib.evaluation.conductance(self.graph_, clust).score
        coverage = cdlib.evaluation.edges_inside(self.graph_, clust).score
        performance = bi_performance(badj, proj0_labels+proj1_labels)
        gini = skbio_gini.gini_index([len(c) for c in communities])

        result = dict(
            name=self.name_,
            num_clusters=(max(graph_labels)+1),
            modularity_score=modularity_score,
            modularity_score_barber=modularity_score_barber,
            modularity_score_murata=modularity_score_murata,
            modularity_score_1=modularity_score_1,
            modularity_score_2=modularity_score_2,
            adj_rand_index=adj_rand_index,
            conductance=conductance,
            coverage=coverage,
            performance=performance,
            gini=gini
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

########################################################
#### Utility
########################################################
def get_best_community_solutions(df_contestants):
    """
    Computes the best solution metrics among the hierarchical communities computed by community detection algorithms
    For a given algorith/graph pair, the best solution maximizes the adjusted rand index score
    """
    assert isinstance(df_contestants, pd.DataFrame), "df_contestants must be a data frame"
    # columns = ['name', 'num_clusters', 'modularity_score', 'modularity_score_1', 'modularity_score_2', 'adj_rand_index', 'graph_idx']
    columns = ['name', 'graph_idx', 'adj_rand_index']
    assert set(columns).issubset(set(df_contestants.columns)), f"Column names must include (in any order): {columns}"
    return df_contestants.groupby(['name', 'graph_idx'])['adj_rand_index'].max().reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

def draw_best_community_solutions(df_best_community_solutions, ax=None):
    """
    Box plots the best solutions (adjusted rand index)
    """
    assert isinstance(df_best_community_solutions, pd.DataFrame), "df_best_community_solutions must be a data frame"
    columns = ['name', 'adj_rand_index']
    assert set(columns).issubset(set(df_best_community_solutions.columns)), f"Column names must include (in any order): {columns}"
    # stats = df_best_community_solutions.groupby(['name'])['adj_rand_index'].describe().reset_index(frop=False)
    # ax = df_best_community_solutions.boxplot(column='adj_rand_index', by='name')
        
    ax = sns.boxplot(y='adj_rand_index', x='name', data=df_best_community_solutions, ax=ax)
    return ax, df_best_community_solutions.groupby(['name'])['adj_rand_index'].describe().reset_index()
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='modularity_score_1', x='name', data=df, ax=axs[1]
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='modularity_score_2', x='name', data=df, ax=axs[2]
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='adj_rand_index', x='name', data=df, ax=axs[3]
    # # , hue=None,
    # )
    # # None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, ax=None, **kwargs)

def bi_performance(badj, communities):
    """
    Calculate the performance of a community assignment, i.e. the fraction of nodes pairs with edges and the same community or without edges and different communities.
    """

    poss_edges = badj.shape[0]*badj.shape[1]
    perf_pairs = 0
    edges = set(zip(badj.tocoo().row,badj.tocoo().col))
    for i in range(badj.shape[0]):
        for j in range(badj.shape[1]):
            if ((i,j) in edges and communities[i] == communities[badj.shape[0]+j]) or ((i,j) not in edges and communities[i] != communities[badj.shape[0]+j]):
                perf_pairs += 1
    return perf_pairs/poss_edges

def modularity_murata(badj,communities):
    """
    Calculate Murata modularity of a given community assignment.
    """
    
    ## Make the e array, fraction of edges between the two communities in each mode.
    e = np.zeros((max(communities)+1,max(communities)+1))
    ## Iterate over the edges.
    for s,t in zip(badj.tocoo().row,badj.tocoo().col):
        ## Increment e_lm where s in comm l and t in comm m.
        e[communities[s]][communities[t+badj.shape[0]]] += 1
    e /= 2*np.sum(e)

    ## Make the a array, the row sums of the e array.
    a = np.sum(e,axis=1)
    
    ## Now we calculate Q, the sum of max observed difference.
    q = 0
    for i in range(e.shape[0]):
        j = np.argmax(e[i])
        q += (e[i][j] - a[i]*a[j])
    return q

def make_badj(graph):
    """
    Turn an igraph object into a biadjency matrix from the edgelist.
    """
    vertex_map = {}  ## Map true id to bipartite id.
    vertex_type = {}
    lid,uid = 0,0
    for v in graph.vs():
        if v['name'] == 1:
            bid = uid
            uid += 1
        else:
            bid = lid
            lid += 1
        vertex_map[v.index] = bid
        vertex_type[v.index] = v['name']
    edge_list = [(e.source,e.target) for e in graph.es]  ## Extract the edges.
    edge_list = [(s,t) if vertex_type[t] else (t,s) for s,t in edge_list]  ## Order them so the bottom node is first.
    edge_list = [(vertex_map[s],vertex_map[t]) for s,t in edge_list]  ## Map them to bipartite ids.
    badj = sknetwork.utils.edgelist2biadjacency(edge_list)  ## Make the adjacency matrix.
    return badj

def test_community_detector():
    # Data generation
    from data_generation import ExpConfig, DataGenerator
    expconfig = ExpConfig()
    expgen = DataGenerator(expconfig=expconfig)
    print(expgen)
    datagen = expgen.generate_data()
    for it in datagen:
        # Save graph somewhere or viz.
        graph = it
        break
    igraph.summary(graph)
    print(graph.is_connected())
    # com_det = ComDetFastGreedy(num_clusters=15)
    # com_det = ComDetEdgeBetweenness(num_clusters=15)
    com_det = ComDetWalkTrap(num_clusters=15)
    print(com_det)
    com_det.detect_communities(graph=graph)
    result = com_det.get_results()
    params = com_det.get_params()
    print(result)
    print(params)
    import pandas as pd
    df = pd.DataFrame(result)
    print(df)

if __name__ == "__main__":
    
    # test_community_detector()
    # Data generation
    from data_generation import ExpConfig, DataGenerator
    expconfig = ExpConfig()
    expgen = DataGenerator(expconfig=expconfig)
    # print(expgen)
    datagen = expgen.generate_data()
    graphs = list(datagen)[:1]
    # igraph.summary(graphs[1])
    # for it in datagen:
    #     # Save graph somewhere or viz.
    #     graph = it
    #     break
    algos = [
        ComDetEdgeBetweenness(max_num_clusters=15),
        ComDetWalkTrap(max_num_clusters=15),
        ComDetFastGreedy(max_num_clusters=15),
    ][:1]
    results = []
    for g_idx, graph in enumerate(graphs):
        # print(g_idx)
        for algo in algos:
            # print(algo)
            result = algo.detect_communities(graph=graph).get_results()
            # print(result)
            for r in result:
                r['graph_idx'] = g_idx
                results.extend(result)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.shape)





    



