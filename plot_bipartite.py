"""Plot a bipartite GML network on two panels, coloured by two node attributes.

Node shape is set by a node "type" attribute. Numeric attributes get a continuous
colormap + colorbar; others get a categorical legend. Edit the settings below.
"""
import math
from numbers import Number

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import igraph as ig

# --- settings ---------------------------------------------------------------
GML_PATH = "moo/dummy_0.gml"
COLOUR_ATTRS = ("GT", "HGT")  # one per panel
TYPE_ATTR = "type"
LAYOUT = "spring"                 # "bipartite" or "spring"
NODE_SIZE = 120
CMAP = "viridis"                     # numeric attributes
CAT_CMAP = "tab10"                   # categorical attributes
# ----------------------------------------------------------------------------

SHAPES = ["o", "s", "^", "D", "v", "p", "h", "*"]


def compute_layout(G, type_attr, layout):
    if layout == "bipartite":
        types = sorted({d.get(type_attr) for _, d in G.nodes(data=True)}, key=str)
        if len(types) == 2:
            top = [n for n, d in G.nodes(data=True) if d.get(type_attr) == types[0]]
            return nx.bipartite_layout(G, top)
    return nx.spring_layout(G, seed=42)


def draw_panel(ax, G, pos, colour_attr, type_attr, shape_map):
    values = [G.nodes[n].get(colour_attr) for n in G.nodes]
    present = [v for v in values if v is not None]
    numeric = False#bool(present) and all(isinstance(v, Number) and not isinstance(v, bool) for v in present)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)

    if numeric:
        vmin, vmax = min(present), max(present)
        colour_of = {n: G.nodes[n].get(colour_attr) for n in G.nodes}
    else:
        cats = sorted({str(v) for v in values})
        cmap = plt.get_cmap(CAT_CMAP)
        cat_colour = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
        colour_of = {n: cat_colour[str(G.nodes[n].get(colour_attr))] for n in G.nodes}

    mappable = None
    for t, shape in shape_map.items():
        nodes = [n for n in G.nodes if G.nodes[n].get(type_attr) == t]
        if not nodes:
            continue
        kw = dict(node_shape=shape, node_size=NODE_SIZE, ax=ax, edgecolors="k", linewidths=0.5)
        if numeric:
            have = [n for n in nodes if colour_of[n] is not None]
            miss = [n for n in nodes if colour_of[n] is None]  # drawn grey
            if have:
                mappable = nx.draw_networkx_nodes(
                    G, pos, nodelist=have, node_color=[colour_of[n] for n in have],
                    cmap=plt.get_cmap(CMAP), vmin=vmin, vmax=vmax, **kw)
            if miss:
                nx.draw_networkx_nodes(G, pos, nodelist=miss, node_color="lightgrey", **kw)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                   node_color=[colour_of[n] for n in nodes], **kw)

    handles = [Line2D([], [], marker=s, ls="", mfc="white", mec="k", ms=8, label=str(t))
               for t, s in shape_map.items()]
    if numeric:
        if mappable is not None:
            plt.colorbar(mappable, ax=ax, label=colour_attr, shrink=0.8)
        ax.legend(handles=handles, title=type_attr, fontsize=8)
    else:
        handles += [Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=8, label=k)
                    for k, c in cat_colour.items()]
        ax.legend(handles=handles, title=f"{type_attr} / {colour_attr}", fontsize=8)

    ax.set_title(colour_attr)
    ax.set_axis_off()


def load_gml(path):
    """Read with igraph, convert to networkx (nodes keyed by igraph index).
 
    igraph fills missing vertex attributes with NaN (numeric) or "" (string);
    these are dropped so they are treated as missing when plotting.
    """
    G = ig.Graph.Read_GML(path).to_networkx()
    for _, d in G.nodes(data=True):
        for k in [k for k, v in d.items() if v == "" or (isinstance(v, float) and math.isnan(v))]:
            del d[k]
    return G
 
 
G = load_gml(GML_PATH)

types = sorted({d.get(TYPE_ATTR) for _, d in G.nodes(data=True)}, key=str)
shape_map = {t: SHAPES[i % len(SHAPES)] for i, t in enumerate(types)}
pos = compute_layout(G, TYPE_ATTR, LAYOUT)  # shared across panels

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, attr in zip(axes, COLOUR_ATTRS):
    draw_panel(ax, G, pos, attr, TYPE_ATTR, shape_map)
fig.tight_layout()
plt.show()