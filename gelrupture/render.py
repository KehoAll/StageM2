import sys, os.path, argparse, shutil, subprocess
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from colloids import povray

    
def simplify_graph(gr):
    """Simplify a graph so that strings of nodes with only two neighbours become edges between clumps of nodes that are either dead ends or have degree higher than 3"""
    #Two subgraph: one with elements having degree 2 and other subgraph with everything else
    notwo = gr.subgraph(
        n
        for n, degree in gr.degree #gr.degree gives an object for iterating over node, degree (n,degree)
        if degree != 2
    )
    onlytwo = gr.subgraph(
        n
        for n, degree in gr.degree
        if degree == 2
    )
    newgraph = nx.Graph()
    #Each clump of skeleton pixels with either 0, 1, 3 or more neighbours is a node.
    for n, clump in enumerate(nx.connected_components(notwo)): #iterating over the connected component object containing set of connected nodes
        #remember the indices of the members of this node
        newgraph.add_node(n, old=clump)
    #Each string of skeleton pixels with two neighbours is an edge.
    for edge in nx.connected_components(onlytwo):
        newedge = []
        #only particles at the ends of the edge can be connected to a clump
        ends = [i for i in edge if onlytwo.degree[i] < 2]
        for end in ends:
            #among the neighbours of this end of the edge in the original graph
            for i in gr[end]:
                #that do not below to the edge itself
                if i not in edge:
                    #look for a clump that contains it
                    for j, clump in newgraph.nodes(data="old"):
                        if i in clump:
                            newedge.append(j)
        #remember the indices of the members of this edge
        if len(newedge):
            if len(ends)==2:
                path = nx.shortest_path(onlytwo, ends[0], ends[1])
            else:
                path = ends
            newgraph.add_edge(newedge[0], newedge[-1], old = path)
        else:
            #case of an isolated closed loop
            n = newgraph.number_of_nodes()
            newgraph.add_node(n, old=edge)
            newgraph.add_edge(n, n, old = [])
    return newgraph  

class SphereSweep(povray.Item):
  def __init__(self, spline="linear_spline", centers=[], radii=[],*opts,**kwargs):
    super().__init__("sphere_sweep",[spline, len(centers)], opts, **kwargs)
    for center, r in zip(centers, radii):
        self.args.append(povray.Vector(list(center)))
        self.args.append(r)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render with POVray a list of stacks from a series of skeletonized xy and bonds.')
    parser.add_argument('prefix', type=str, help='path and prefix of the input and output file names')
    parser.add_argument('n', type=int, help='Number of time steps that should be used.')
    parser.add_argument('--rmin', type=float, default=4.43, help='Minimum size of the spheres.')
    parser.add_argument('--npix', type=int, default=512, help='Number of pixels in the exported images.')
    args = parser.parse_args()


    for t in range(args.n):
        importname = "{}_squeleton_t{:03d}".format(args.prefix, t)
        pos = np.loadtxt(importname+".xy")
        bonds = np.loadtxt(importname+".bonds", dtype=int)
        
        #create graph
        gr = nx.Graph()
        gr.add_nodes_from(np.arange(len(pos)))
        gr.add_edges_from(bonds)
        #simplify graph
        sgr = simplify_graph(gr)
        
        exportname = "{}_t{:03d}".format(args.prefix, t)
        
        f = povray.File(exportname+".pov", "colors.inc", "shear.inc")
        clumps = [
            povray.Sphere(
                pos[list(data['old'])].mean(0).tolist(), 
                max(np.sqrt(pos[list(data['old'])].std(0).sum()), args.rmin))
            for clump, data in sgr.nodes.items()
            if sgr.degree(clump)>1
            ]
        clumps.append(povray.Texture(povray.Pigment(color="Red")))
        edges = [
            SphereSweep(
                "b_spline",
                pos[list(data['old'])], 
                np.full(len(data['old']), args.rmin)
                )
            for edge, data in sgr.edges.items()
            if len(data['old'])>3
            ]
        edges += [
            SphereSweep(
                "linear_spline",
                pos[list(data['old'])], 
                np.full(len(data['old']), args.rmin)
                )
            for edge, data in sgr.edges.items()
            if len(data['old']) in [2,3]
            ]
        edges += [
            povray.Sphere((x,y,z), args.rmin)
            for edge, data in sgr.edges.items()
            for x,y,z in pos[list(data['old'])]
            if len(data['old'])==1
            ]
        #edges = [
         #   povray.Sphere((x,y,z), args.rmin)
          #  for edge, data in sgr.edges.items()
           # for x,y,z in pos[list(data['old'])]
            #]
        edges.append(povray.Texture(povray.Pigment(color="Gray")))
        ends = [
            povray.Sphere(
                pos[list(data['old'])].mean(0).tolist(), 
                max(np.sqrt(pos[list(data['old'])].std(0).sum()), args.rmin))
            for clump, data in sgr.nodes.items()
            if sgr.degree(clump)==1
            ]
        ends.append(povray.Texture(povray.Pigment(color="White")))
        f.write(povray.Union(*clumps), povray.Union(*ends), povray.Union(*edges))
        f.file.flush()
        #call povray and 
        subprocess.check_call(['povray', '-O%s'%(exportname+'.png'), '+W%d'%args.npix, '+H%d'%args.npix, '-D00' , exportname+'.pov'])
        plt.imsave(exportname+'.jpg', plt.imread(exportname+'.png'))
        os.remove(exportname+'.png')
