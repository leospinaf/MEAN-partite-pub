import numpy as np
import igraph as ig

from biSBM.optimalks import OptimalKs
from biSBM.utils import assemble_old2new_mapping, assemble_edgelist_old2new
from engines.kl import KL

# --- 1. a toy bipartite graph in igraph ---
# 4 nodes of type-a (indices 0-3), 3 nodes of type-b (indices 4-6)
g = ig.Graph.Bipartite(
    types=[0, 0, 0, 0, 1, 1, 1],
    edges=[(0, 4), (0, 5), (1, 4), (1, 6), (2, 5), (2, 6), (3, 4), (3, 6)],
)

# --- 2. build the types array bipartiteSBM expects: 1 = type-a, 2 = type-b ---
old_types = np.array([1 if not t else 2 for t in g.vs["type"]])
old_edgelist = np.array(g.get_edgelist())

# --- 3. reorder nodes so type-a block comes first, contiguously ---
old2new, new2old, new_types = assemble_old2new_mapping(old_types)
new_edgelist = assemble_edgelist_old2new(old_edgelist, old2new)

na = int(np.sum(new_types == 1))
nb = int(np.sum(new_types == 2))

# --- 4. set up the Kernighan-Lin engine ---
kl = KL(
    f_engine="engines/bipartiteSBM-KL/biSBM",
    n_sweeps=2,
    is_parallel=False,
    n_cores=1,
    kl_edgelist_delimiter="\t",
    kl_steps=4,
    kl_itertimes=1,
    f_kl_output="engines/bipartiteSBM-KL/f_kl_output",
)

# --- 5. run model selection to find the best (Ka, Kb) ---
oks = OptimalKs(kl, new_edgelist, new_types)
oks.set_params(init_ka=2, init_kb=2, i_0=0.1)
oks.minimize_bisbm_dl()

print(oks.summary())

# --- 6. get the actual partition (membership vector) at the best (Ka, Kb) ---
best_ka, best_kb = ...  # read these off oks.summary()
dl, e_rs, mb = oks.compute_dl(best_ka, best_kb)
print("description length:", dl)
print("membership vector (new-index order):", mb)

# map the membership vector back to your original igraph vertex ids
mb_original_order = np.zeros(len(mb), dtype=int)
for new_id, label in enumerate(mb):
    mb_original_order[new2old[new_id]] = label
print("membership vector (original igraph order):", mb_original_order)