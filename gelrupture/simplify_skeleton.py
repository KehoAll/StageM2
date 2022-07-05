import sys, os.path, argparse
from datetime import datetime
import numpy as np
from scipy.ndimage import binary_dilation, convolve, measurements
import networkx as nx
import check
import h5py,tables
import trackpy as tp
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a binary skeleton in a h5 file into a simplified network of nodes compatible with trackpy.')
    parser.add_argument('filename', type= str, help = 'path and name of the h5 file')
    parser.add_argument('-o','--output',default='simple_skeleton.h5', type=str, help = 'output path and file name')
    parser.add_argument('--no_clean_git', action='store_true', help='Bypass checking wether my git repository is clean (Use only for tests).')
    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)
    if not args.no_clean_git:
        args.commit = check.clean_git(__file__).hexsha
    args.datetime = datetime.now().isoformat()

    linked_position_type = np.dtype([("tr", np.int32), ("xyz", np.float64, 3)])

    #open the input h5 file in read only using h5py
    #open the output h5 file in append mode using pytables through trackpy and pandas
    with h5py.File(args.filename, "r") as input, tp.PandasHDFStore("simple_skeleton.h5", "a") as output:
        px = input.get("voxel_size_in_microns")[:]
        skb = input.get("binary_skeleton")
        bottom = skb.attrs["bottom"]

        #save all analysis parameters
        for k,v in args.__dict__.items():
            output.store._handle.root._v_attrs[k] = v
        #preparer the group in which to store bonds
        bondgrp = output.store._handle.create_group("/", "bonds")

        #define neighborhood as a 3x3x3 cube
        structure = np.ones((3,3,3), np.uint8)

        for t in range(skb.shape[0]):
            fr_sk = skb[t]
            #number of neighbours
            Nngb = convolve(fr_sk.astype(np.uint8), structure, mode='constant')*fr_sk - fr_sk
            #Groups of continuous pixels that have 1 or more than three neighbours are nodes of the simplified graph. Label them
            notwo = (Nngb>0) & (Nngb!=2)
            labeled_newnodes,Nnewnodes = measurements.label(notwo, structure)
            if Nnewnodes==0:
                print("Empty skeleton at t=%d"%t)
                continue
            #Create the new graph and populate its nodes. Stores the position of the center of mass.
            newgraph = nx.Graph()
            for n, c in enumerate(measurements.center_of_mass(fr_sk, labeled_newnodes, np.arange(Nnewnodes)+1)):
                newgraph.add_node(n, pos=c)
            #Groups of contiguous pixels that have exactly two neighbours are edges of the simplified graph. Label them
            onlytwo = Nngb==2
            labeled_newedges,Nnewedges = measurements.label(onlytwo, structure)
            #Find which nodes each edge is connecting
            #Get the subvolume that exactly contains each edge
            for l, slices in enumerate(measurements.find_objects(labeled_newedges)):
                #Extend this subvolume by one pixel, managing edges of the picture
                extended_slices = tuple(slice(max(0, s.start-1), min(s.stop+1, L), None) for s,L in zip(slices, fr_sk.shape))
                #in this subvolume, select only the current edge and dilate it
                dilated_edge = binary_dilation(labeled_newedges[extended_slices]==l+1, structure)
                #find which new node is intersecting the dilated edge
                neighbourhood = labeled_newnodes[extended_slices][dilated_edge]
                newedge = np.sort(neighbourhood[neighbourhood>0]) -1
                if len(newedge):
                    #case of an edge that connects two nodes
                    newgraph.add_edge(newedge[0], newedge[-1])
                else:
                    #case of an isolated closed loop
                    n = newgraph.number_of_nodes()
                    c = measurements.center_of_mass(fr_sk, labeled_newedges, l+1)
                    newgraph.add_node(n, pos=c)
                    newgraph.add_edge(n, n)
            #save the resulting graph
            coords = pd.DataFrame(
                (np.array([data["pos"] for node, data in newgraph.nodes.items()]) + [bottom, 0, 0])* px[::-1],
                columns=tp.utils.default_pos_columns(fr_sk.ndim)
            )
            coords['frame'] = t
            output.put(coords)
            output.store._handle.create_array(
                bondgrp,
                tp.framewise_data.code_key(t),
                np.array([edge for edge, data in newgraph.edges.items()], dtype=np.int32)
            )
            print('\r%d (%.01f %%)'%(t, 100*(t+1)/skb.shape[0]), sep=' ', end=' ', flush=True)
        print("\ndone!")
