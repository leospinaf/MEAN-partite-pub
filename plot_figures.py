## This script plots the various figures required for the BipartiteMOEA paper.

import code
import igraph
from moo.data_generation import ExpConfig, DataGenerator
from igraph import *
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patches as mpatches

## Better plotting behvaiour:
def make_int_col(rgb):
	return tuple(map(int,map(lambda x: x*256, rgb)))

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
blue, green, red, purple, white  = (0.368, 0.506, 0.709), (0.560, 0.691, 0.194), (0.922, 0.385, 0.209), (0.528, 0.470, 0.701), (1,1,1)


## Figure 1: A simple bipartite with two communities.
expconfig = ExpConfig(
	L=[3,2],U=[2,3],NumEdges=10,BC=0.05,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=13 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = ['gold' if g else red for g in groundtruth]

	g3.vs["shape"] = shapes

## Plot using igraph.
#plot(g3,"figs/new.demo.normal.pdf",vertex_label=["1", "2", "3","4","5","6","7","8","9","10"],vertex_label_size=25,vertex_size=35)

## Plot using networkx.
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=550)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=550)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,font_color='black',labels={n:n+1 for n in G},font_weight='bold',font_size=14)
plt.axis('off')
plt.savefig('figs/new.demo.normal.pdf',bbox_inches='tight')
plt.close()

## Figure 2: The unipartite projections of the example in Figure 1.
g1, g2 = g3.bipartite_projection(multiplicity=True)
g1.es["width"] = 2**np.array(g1.es["weight"])-1
g2.es["width"] = 2**np.array(g2.es["weight"])-1
#code.interact(local=locals())
#g4 = igraph.union([g1,g2])

#plot(g1,"figs/new.demo.proj1.original.pdf",vertex_label=["4", "5", "6","8","10"],vertex_label_size=25,vertex_size=45,margin=50)
G = g1.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=550)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=550)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[G.edges[e]['width'] for e in G.edges])
nx.draw_networkx_labels(G,pos,font_color='black',labels={n:l for n,l in zip(G,["4", "5", "6","8","10"])},font_weight='bold',font_size=14)
plt.axis('off')
plt.savefig('figs/new.demo.proj1.original.pdf',bbox_inches='tight')
plt.close()

#plot(g2,"figs/new.demo.proj2.original.pdf",vertex_label=["1", "2", "3","7","9"],vertex_label_size=25,vertex_size=45,margin=50)
G = g2.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=550)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=550)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[G.edges[e]['width'] for e in G.edges])
nx.draw_networkx_labels(G,pos,font_color='black',labels={n:l for n,l in zip(G,["1", "2", "3","7","9"])},font_weight='bold',font_size=14)
plt.axis('off')
plt.savefig('figs/new.demo.proj2.original.pdf',bbox_inches='tight')
plt.close()

## Figure 3: Edge and node centrality for bipartite example.
edge_betweenness=g3.edge_betweenness(directed=False)
g3.es['betweenness'] = edge_betweenness
g3.es['curved'] = False
vertex_betweenness=g3.betweenness(directed=False)
#plot(g3,"figs/new.demo.normal.centrality.pdf",vertex_label=vertex_betweenness,vertex_label_size=15,vertex_size=35,edge_label=edge_betweenness,edge_label_size=20)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=550)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=550)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,font_color='black',labels={i:n for i,n in enumerate(vertex_betweenness)},font_weight='bold',font_size=12)
nx.draw_networkx_edge_labels(G,pos,font_color='black',edge_labels={e:G.edges[e]['betweenness'] for e in G.edges},font_weight='bold',font_size=12)
plt.axis('off')
plt.savefig('figs/new.demo.normal.centrality.pdf',bbox_inches='tight')
plt.close()

## Figure 4: Minimum spanning tree from the bipartite example.
weights=g3.edge_betweenness()
t = g3.spanning_tree(weights)

#plot(t,"figs/new.demo.mst.normal.pdf",vertex_label=["1", "2", "3","4","5","6","7","8","9","10"],vertex_label_size=25,vertex_size=35)
G = t.to_networkx()
#pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=550)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=550)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,font_color='black',labels={n:n+1 for n in G},font_weight='bold',font_size=14)
plt.axis('off')
plt.savefig('figs/new.demo.mst.normal.pdf',bbox_inches='tight')
plt.close()

## Figure 5: Schematic of the adjacency representation.
# Figure is a series of panels - arrange as a subfigure.
#fig,ax = plt.subplots(2,4,figsize=(12,6),gridspec_kw={'width_ratios':[2,2,3,2]})  ## Grid of two rows and four columns.
fig,ax = plt.subplots(1,4,figsize=(12,3),gridspec_kw={'width_ratios':[2,2,3,2]})  ## Grid of two rows and four columns.
fig.tight_layout()
col_map = {1:'gold',0:red}

## The representation table.
tab = ax[0].table(cellText=[['1','1'],['2','1'],['3','3'],['4','0'],['5','1'],['6','1'],['7','1'],['8','0'],['9','1'],['10','1']],cellColours=[['white',(233/255,235/255,245/255)]]*10,loc='center',colWidths=[0.1,0.1])
tab.scale(1.1,1.2)
for key,cell in tab.get_celld().items():
	cell.set_edgecolor('white')
ax[0].axis('off')
ax[0].annotate('Representation',xy=(0.33,0.93))

## The adjacency list.
tab=ax[1].table(cellText=[['1','8','10','',''],['2','4','6','',''],['3','4','5','6','10'],['4','2','3','',''],
	['5','3','','',''],['6','2','3','',''],['7','10','','',''],['8','1','','',''],['9','10','','',''],['10','1','3','7','9']]
	,loc='center',colWidths=[0.1,0.1,0.1,0.1,0.1],
	cellColours=[['white',(233/255,235/255,245/255),(233/255,235/255,245/255),(233/255,235/255,245/255),(233/255,235/255,245/255)]]*10)
tab.scale(1.1,1.2)
ax[1].axis('off')
ax[1].annotate('Adjacency list',xy=(0.33,0.93))

## Colour the relevant text.
tab[(0,1)].get_text().set_color(red)
tab[(1,1)].get_text().set_color(red)
tab[(2,3)].get_text().set_color(red)
tab[(4,1)].get_text().set_color(red)
tab[(5,1)].get_text().set_color(red)
tab[(6,1)].get_text().set_color(red)
tab[(8,1)].get_text().set_color(red)
tab[(9,1)].get_text().set_color(red)

for key,cell in tab.get_celld().items():
	cell.set_edgecolor('white')

## The graph representation.
gr = igraph.Graph()
gr.add_vertices(10)
gr.vs['name'] = [str(i+1) for i,_ in enumerate(gr.vs)]
gr.vs['shape'] = ['s','s','s','o','o','o','s','o','s','o']
gr.vs['part'] = [1,0,0,0,0,0,1,1,1,1]
gr.add_edges([(3,1),(1,5),(5,2),(2,4),(7,0),(0,9),(9,6),(9,8)])

G = gr.to_networkx()
#pos = nx.kamada_kawai_layout(G)
pos = {0:(6.5,7),1:(0,7),2:(0,3.5),3:(1,9),4:(1.5,2),5:(-1,5),6:(9.5,5),7:(6,9.5),8:(6.5,2),9:(7.5,4.5)}
nodes=nx.draw_networkx_nodes(G,pos,ax=ax[2],node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='o'],node_color=[col_map[G.nodes[n]['part']] for n in G if G.nodes[n]['shape']=='o'],node_size=300)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,ax=ax[2],node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='s'],node_color=[col_map[G.nodes[n]['part']] for n in G if G.nodes[n]['shape']=='s'],node_size=300)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,ax=ax[2])
nx.draw_networkx_labels(G,pos,ax=ax[2],font_color='black',labels={n:G.nodes[n]['name'] for n in G},font_weight='bold',font_size=14)

ax[2].axis('off')
ax[2].annotate('Graph representation',xy=(0.33,10.8))

## Circle the components.
e1 = mpatches.Ellipse((0.7,5.4),6,9,fill=False,edgecolor='black')
ax[2].add_patch(e1)
e2 = mpatches.Ellipse((7.3,5.6),6.5,11,fill=False,edgecolor='black')
ax[2].add_patch(e2)
ax[2].autoscale_view()

## The membership vector.
tab = ax[3].table(cellText=[['1','1'],['2','2'],['3','2'],['4','2'],['5','2'],['6','2'],['7','1'],['8','1'],['9','1'],['10','1']],cellColours=[['white',(233/255,235/255,245/255)]]*10,loc='center',colWidths=[0.1,0.1])
tab.scale(1.1,1.2)
for key,cell in tab.get_celld().items():
	cell.set_edgecolor('white')
ax[3].axis('off')
ax[3].annotate('Membership vector',xy=(0.33,0.93))

## The input graph.
gr = igraph.Graph()
gr.add_vertices(10)
gr.vs['name'] = [str(i+1) for i,_ in enumerate(gr.vs)]
gr.vs['shape'] = ['s','s','s','o','o','o','s','o','s','o']
gr.vs['part'] = [1,0,0,0,0,0,1,1,1,1]
gr.add_edges([(3,1),(1,5),(5,2),(2,4),(7,0),(0,9),(9,6),(9,8),(2,3),(2,9)])

G = gr.to_networkx()
pos = nx.kamada_kawai_layout(G)
#nodes=nx.draw_networkx_nodes(G,pos,ax=ax[1,1],node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='o'],node_color=[col_map[G.nodes[n]['part']] for n in G if G.nodes[n]['shape']=='o'],node_size=300)
#nodes.set_edgecolor('black')
#nodes = nx.draw_networkx_nodes(G,pos,ax=ax[1,1],node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='s'],node_color=[col_map[G.nodes[n]['part']] for n in G if G.nodes[n]['shape']=='s'],node_size=300)
#nodes.set_edgecolor('black')
#nx.draw_networkx_edges(G,pos,ax=ax[1,1])
#nx.draw_networkx_labels(G,pos,ax=ax[1,1],font_color='black',labels={n:G.nodes[n]['name'] for n in G},font_weight='bold',font_size=14)

#ax[1,1].axis('off')

## Adjust the spacing between subfig panels.
fig.subplots_adjust(wspace=0.05)

## Add in the arrows between panels.
arr = mpatches.ConnectionPatch(xyA=(0.9,0.5),coordsA=ax[0].transData,
								xyB=(0.1,0.5),coordsB=ax[1].transData,
								arrowstyle='->')
fig.add_artist(arr)
arr = mpatches.ConnectionPatch(xyA=(0.9,0.5),coordsA=ax[1].transData,
								xyB=(-3,5.6),coordsB=ax[2].transData,
								arrowstyle='->')
fig.add_artist(arr)
arr = mpatches.ConnectionPatch(xyA=(12,5.6),coordsA=ax[2].transData,
								xyB=(0.3,0.5),coordsB=ax[3].transData,
								arrowstyle='->')
fig.add_artist(arr)
#arr = mpatches.ConnectionPatch(xyA=(0.075,1.2),coordsA=ax[1,1].transData,
#								xyB=(0.5,0),coordsB=ax[0,1].transData,
#								arrowstyle='->')
#fig.add_artist(arr)

#ax[1,0].axis('off')
#ax[1,2].axis('off')
#ax[1,3].axis('off')

plt.savefig('figs/new.representation.pdf',bbox_inches='tight')
plt.close()

## Figure 6: Integration of heuristic bias.
# Left panel: Generate a graph with 4 communities.
colour_map = {0:red,1:blue,2:green,3:'gold'}
expconfig = ExpConfig(
	L=[10,10,10,10],U=[10,10,10,10],NumEdges=100,BC=0.15,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=13 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes
layout = g3.layout('kk')

#plot(g3,"figs/new.k4.normal.pdf",vertex_size=20,layout=layout)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.k4.normal.pdf',pad_inches=-0.2,bbox_inches='tight')
plt.close()

# Centre panel: adjust edge width by centrality and vertex size by mutation probability.
edge_centrality = g3.edge_betweenness()
g3.es['betweenness'] = edge_centrality
node_centrality = g3.betweenness()
node_degree = g3.degree()
mut_prob = [(b/d)/(sum(node_centrality)) for b,d in zip(node_centrality,node_degree)]
g3.vs['mut_prob'] = mut_prob

G = g3.to_networkx()
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=[10+(1500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='circle'])
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=[10+(1500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='triangle'])
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[0.01*G.edges[e]['betweenness'] for e in G.edges])
plt.axis('off')
plt.savefig('figs/new.k4.w2.normal.pdf',pad_inches=-0.2,bbox_inches='tight')
plt.close()

# Right panel: Plot the MST with the same layout used for the other panels.
weights=g3.edge_betweenness()
t = g3.spanning_tree(weights)

G = t.to_networkx()
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=[10+(1500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='circle'])
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=[10+(1500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='triangle'])
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[0.01*G.edges[e]['betweenness'] for e in G.edges])
plt.axis('off')
plt.savefig('figs/new.k4.init4normal.pdf',pad_inches=-0.2,bbox_inches='tight')
plt.close()

## Figure 7: Examples of unipartite projections from example in Figure 6.
g1, g2 = g3.bipartite_projection(multiplicity=True)
g1.es["width"] = g1.es["weight"]
g2.es["width"] = g2.es["weight"]

#plot(g1,"figs/new.k4.proj1.pdf",vertex_size=20)
G = g1.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=[15+(2500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='circle'])
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=[15+(2500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='triangle'])
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[G.edges[e]['width'] for e in G.edges])
plt.axis('off')
plt.savefig('figs/new.k4.proj1.pdf',pad_inches=-0.2,bbox_inches='tight')
plt.close()

#plot(g2,"figs/new.k4.proj2.pdf",vertex_size=20)
G = g2.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=[15+(2500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='circle'])
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=[15+(2500*G.nodes[n]['mut_prob']) for n in G.nodes if G.nodes[n]['shape']=='triangle'])
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos,width=[G.edges[e]['width'] for e in G.edges])
plt.axis('off')
plt.savefig('figs/new.k4.proj2.pdf',pad_inches=-0.2,bbox_inches='tight')
plt.close()

## Figure 8: Examples with generator parameter variations.
# Panel 1 - simple graph.
expconfig = ExpConfig(
	L=[10,10,10,10],U=[10,10,10,10],NumEdges=100,BC=0.15,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=13 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes

#plot(g3,"figs/new.SimpleGraph.pdf",vertex_size=25)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.SimpleGraph.pdf',bbox_inches='tight')
plt.close()

# Panel 2 - simple graph with mode size imbalance.
expconfig = ExpConfig(
	L=[15,15,15,15],U=[5,5,5,5],NumEdges=100,BC=0.15,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=13 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes

#plot(g3,"figs/new.SimpleGraphImbalance.pdf",vertex_size=25)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.SimpleGraphImbalance.pdf',bbox_inches='tight')
plt.close()

# Panel 3 - simple graph with increased noise.
expconfig = ExpConfig(
	L=[10,10,10,10],U=[10,10,10,10],NumEdges=100,BC=0.4,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=13 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes

#plot(g3,"figs/new.SimpleGraphNoise.pdf",vertex_size=25)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.SimpleGraphNoise.pdf',bbox_inches='tight')
plt.close()

# Panel 4 - complex graph with multiple deviations.
expconfig = ExpConfig(
	L=[15,15,15,15],U=[5,5,5,5],NumEdges=100,BC=0.4,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=15 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes

#plot(g3,"figs/new.ComplexGraph.pdf",vertex_size=25)
G = g3.to_networkx()
pos = nx.kamada_kawai_layout(G)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.ComplexGraph.pdf',bbox_inches='tight')
plt.close()

## Figure 9 - redraw the graph with new scheme.
expconfig = ExpConfig(
	L=[50,25,10,5], U=[5,5,5,5], NumEdges=220, BC=0.2,NumGraphs=1,
	shuffle=True, # Shuffle labels (or no)
	seed=42 # 3 35 For reproducibility (this is the default, but can be changed)
	)

expgen = DataGenerator(expconfig=expconfig)
datagen = expgen.generate_data()

for g_idx, graph in enumerate(datagen):

	g3 = graph
	groundtruth = g3.vs['GT']
	vertices = g3.vs['VX']
	# Update shape vector
	shapes = []
	color = []
	for i in range(0,len(vertices)):
		if vertices[i]==1:
			shapes += ["triangle"]
		else:
			shapes += ["circle"]

	g3.vs["color"] = g3.vs["color"] = [colour_map[g] for g in groundtruth]

	g3.vs["shape"] = shapes

#plot(g3,"figs/new.ComplexGraph.pdf",vertex_size=25)
G = g3.to_networkx()
pos = nx.spring_layout(G,seed=43)#,k=0.1)
nodes=nx.draw_networkx_nodes(G,pos,node_shape='o',nodelist=[n for n in G if G.nodes[n]['shape']=='circle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='circle'],node_size=150)
nodes.set_edgecolor('black')
nodes = nx.draw_networkx_nodes(G,pos,node_shape='s',nodelist=[n for n in G if G.nodes[n]['shape']=='triangle'],node_color=[G.nodes[n]['color'] for n in G if G.nodes[n]['shape']=='triangle'],node_size=150)
nodes.set_edgecolor('black')
nx.draw_networkx_edges(G,pos)
plt.axis('off')
plt.savefig('figs/new.example.pdf',bbox_inches='tight')
plt.close()