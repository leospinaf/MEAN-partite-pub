import numpy as np
import igraph as ig
from sparsebm import LBM
from sparsebm import ModelSelection

def make_badj(graph):
    """
    Turn an igraph object into a biadjency matrix from the edgelist.
    """
    vertex_map = {}  ## Map true id to bipartite id.
    vertex_type = {}
    lid,uid = 0,0
    for v in graph.vs():
        if v['type'] == 1:
            bid = uid
            uid += 1
        else:
            bid = lid
            lid += 1
        vertex_map[v.index] = bid
        vertex_type[v.index] = v['type']
    edge_list = [(e.source,e.target) for e in graph.es]  ## Extract the edges.
    edge_list = [(s,t) if vertex_type[t] else (t,s) for s,t in edge_list]  ## Order them so the bottom node is first.
    edge_list = [(vertex_map[s],vertex_map[t]) for s,t in edge_list]  ## Map them to bipartite ids.
    badj = edgelist_to_biadjacency(edge_list)  ## Make the adjacency matrix.
    return badj

from scipy import sparse

def edgelist_to_biadjacency(edges, shape=None):
    rows = np.array([e[0] for e in edges])
    cols = np.array([e[1] for e in edges])

    if shape is None:
        shape = (rows.max() + 1, cols.max() + 1)

    data = np.ones(len(edges))
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)

# --- 1. the same toy bipartite graph as before ---
g = ig.Graph.Bipartite(
    types=[0, 0, 0, 0, 1, 1, 1],
    edges=[(0, 4), (0, 5), (1, 4), (1, 6), (2, 5), (2, 6), (3, 4), (3, 6)],
)

# --- 2. get the biadjacency matrix directly — no reordering needed ---
X = make_badj(g)  # shape: (n_type_a_nodes, n_type_b_nodes)

# --- 3a. fit LBM with a known number of row/column clusters ---
model = LBM(
    n_row_clusters=2,
    n_column_clusters=2,
    n_init_total_run=5,
    verbosity=1,
)
model.fit(X)

print("Row labels (type-a communities):", model.row_labels)
print("Column labels (type-b communities):", model.column_labels)
print("ICL (model fit criterion):", model.get_ICL())

# --- 3b. or let it search over the number of clusters automatically ---
# (this is the direct analogue of bipartiteSBM's Ka/Kb search)
model_selection = ModelSelection("LBM")
models = model_selection.fit(X)

print("Best row labels:", models.best.row_labels)
print("Best column labels:", models.best.column_labels)