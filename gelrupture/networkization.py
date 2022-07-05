import numpy as np
from colloids import lif
from matplotlib import pyplot as plt
from colloids import lif
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage import morphology
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import cKDTree as KDTree
from pims import Bioformats, pipeline
import networkx as nx

#See numLoop_Mathieu_190628 for information on the code

def skeletonize (frame, bottom, top, cut,bi_crit):
# Give the skeletonize binary array (image) for a frame. Returns pos which is the resultant array of binarization of the section of the frame. Sectioning is decided by bottom, top and cut
    fr = frame[bottom:top]
    profile = fr.mean(-1).mean(-1)[:,None,None]
    bg = gaussian_filter(fr.mean(0),25)
    radius = 5
    fr_gus = gaussian_filter(fr,5)
    fr_norm = fr_gus / bg / profile
    fr_bi = fr_norm > bi_crit * filters.threshold_mean(fr_norm)
    fr_sk = morphology.skeletonize_3d(fr_bi)
    L = fr_sk.shape
    fr_sk_cut = fr_sk[1:L[0]-1,cut+1:L[1]-cut-1, cut+1:L[2]-cut-1]
    pos = np.vstack(np.where(fr_sk_cut>0)).T
    return pos

def graph(pos, leaf_size, maxbondlength):
    tree = KDTree(pos,leaf_size)
    maxbondlength = maxbondlength
    bonds = tree.query_pairs(maxbondlength, output_type = 'ndarray')
    gr = nx.Graph()
    gr.add_nodes_from(np.arange(len(pos)))
    gr.add_edges_from(bonds)
    return gr
    

def skeletonVTK(filename, pos, gr):
    file = open(filename + ".vtk",'w')
    #
    file.write("# vtk DataFile Version 2.0 \n")
    file.write("Field Emission Device - Charge Density Plot  \n")
    file.write("ASCII  \n")
    # positions of the particles
    file.write("DATASET POLYDATA  \n")
    file.write("POINTS " + str(pos.shape[0]) + "float \n" )
    np.savetxt(file, pos, fmt='%d', delimiter=" ", newline="\n")
    file.write('LINES %i %i\n' % (gr.number_of_edges(), 3*gr.number_of_edges()))
    np.savetxt(
        file, 
        np.column_stack((np.full(gr.number_of_edges(), 2), gr.edges)),
        fmt='%d', delimiter=" ", newline="\n")
    file.write("POINT_DATA " + str(pos.shape[0]) + "\n" )
    file.write("SCALARS neighbour float 1 \n" )
    file.write("LOOKUP_TABLE default \n" )
    for n, degree in gr.degree:
        file.write('%d\n'%degree)
    file.close()
    
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
            newgraph.add_edge(newedge[0], newedge[-1], old = edge)
        else:
            #case of an isolated closed loop
            n = newgraph.number_of_nodes()
            newgraph.add_node(n, old=edge)
            newgraph.add_edge(n, n, old = [])
    return newgraph  
    
def clumping(sgr):
    pos_clump = np.vstack([pos[list(clump)].mean(0) for j,clump in sgr.nodes(data = 'old')])
    return pos_clump

def count_holes(sgr):
    NCC = sum(1 for i in nx.connected_components(sgr))
    return np.sum((np.array([degree for n, degree in sgr.degree])-2)/2) + NCC

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Derive the skeleton of the gel, built the network graph and give the vtk file of it and count the number of holes')
    parser.add_argument('xyzt', type= bool, help = 'True if the aquitision is a xyz time series else false')
    parser.add_argument('t', type = int, help = 'The time lapse till which we do networkization analysis')
    parser.add_argument('lif_dir', help='Location of the lif image file')
    parser.add_argument('series', type=int, help= 'The 3d stack or the 3d time series ')
    parser.add_argument('bottom', type=int, help='The bottom plane of the stack to pe processsed')
    parser.add_argument('top', type=int, help='The top plane of the stack to pe processsed')
    parser.add_argument('dir', help='The location where the results are stored')
    parser.add_argument('shear', help='The applied shear')
    parser.add_argument('--cut','--cut',default=383, type=int, help = 'The size of the plane is decided by the cut value')
    parser.add_argument('--lf','--leaf_size',type=int, default=12, help='the leafsize of the graph being constructed')
    parser.add_argument('--mbl','--maxbondlength', default=1.8, type=float, help='The maximum edge length or distance in pixel to define a neighbour in the kd tree')
    parser.add_argument('--bi_ct','--bi_crit', default=1.2, type=float, help='The criteria for binarizing the pixels')

    args=parser.parse_args()
    if args.xyzt == True:
        r = lif.Reader(args.lif_dir)
        s = r.getSeries()[args.series]
        frames = Bioformats(args.lif_dir, series = args.series)
        holes = []
        for i in range(0,args.t):
            fr = frames[i]
            voxelSizes = np.array([s.getVoxelSize(d) for d in range(1,4)])[::-1]
            pos = skeletonize(fr, bottom = args.bottom, top=args.top, cut=args.cut, bi_crit = args.bi_ct)
            gr = graph(pos, leaf_size = args.lf, maxbondlength= args.mbl)
            cc = list(nx.connected_components(gr))
            sgr = simplify_graph(gr)
            pos_clump = clumping(sgr)
            skeletonVTK(args.dir+ "NetworkScaled_shear_%s"%i , pos*voxelSizes*1e6, gr)
            skeletonVTK(args.dir+ "SimplifiedNetworkScaled_shear_%s"%i , pos_clump*voxelSizes*1e6, sgr)
            hol = count_holes(sgr)
            holes.append(hol)
    else:    
        r = lif.Reader(args.lif_dir)
        s= r.getSeries()[args.series]
        fr= s.getFrame()
        voxelSizes = np.array([s.getVoxelSize(d) for d in range(1,4)])[::-1]
        pos = skeletonize(fr, bottom = args.bottom, top=args.top, cut=args.cut, bi_crit = args.bi_ct)
        gr = graph(pos, leaf_size = args.lf, maxbondlength= args.mbl)
        cc = list(nx.connected_components(gr))
        sgr = simplify_graph(gr)
        pos_clump = clumping(sgr)
        skeletonVTK(args.dir+ "NetworkScaled_shear"+ args.shear, pos*voxelSizes*1e6, gr)
        skeletonVTK(args.dir+ "SimplifiedNetworkScaled_shear"+ args.shear, pos_clump*voxelSizes*1e6, sgr)
        holes = count_holes(sgr)
    print(holes)


