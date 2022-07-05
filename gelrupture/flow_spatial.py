import os, os.path
import argparse
from datetime import datetime
import numpy as np
import check
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the gradient of the dispacements.')
    parser.add_argument('filename', type= str, help = 'path and name of the h5 file that contains the flow data')
    parser.add_argument('--Zfactor',type=float, default=1., help='Multiplicative factor on Z distances, e.g. due to refractive index corrections.')
    parser.add_argument('--no_clean_git', action='store_true', help='Bypass checking wether my git repository is clean (Use only for tests).')
    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)
    if not args.no_clean_git:
        args.commit = check.clean_git(__file__).hexsha
    args.datetime = datetime.now().isoformat()

    #prepare output
    with h5py.File(args.filename, "a") as h5file:
        flow = h5file.get("FlowField3D")
        if "voxel_size_in_microns" not in h5file:
            import read_lif as lif
            ser = lif.Reader(flow.attrs["filename"]).getSeries()[flow.attrs["series"]]
            #voxel size in XYZ order (microns)
            px = np.asarray([ser.getVoxelSize(d+1)*1e6 for d in range(3)])
            px[-1] *= args.Zfactor
            px = h5file.create_dataset(
                "voxel_size_in_microns",
                data=px,
                shape=(3,), dtype=np.float64
                )
        if "binning_size_in_pixels" not in h5file:
            bins = h5file.require_dataset(
            "binning_size_in_pixels",
            data= np.array([flow.attrs["binning"], flow.attrs["binning"],1]),
            shape=(3,), dtype=np.float64)
        else:
            px = h5file.get("voxel_size_in_microns")
            bins = h5file.get("binning_size_in_pixels")
        gradient = h5file.require_dataset(
            "gradient_flow",
            (flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3], flow.shape[4], flow.shape[4]),
            dtype=np.float32)
        #save all analysis parameters
        for k,v in args.__dict__.items():
            gradient.attrs[k] = v
        for t in range(flow.shape[0]):
            for i in range(gradient.shape[-2]):
                fl = flow[t,...,i]*px[i]
                for j,g in enumerate(np.gradient(fl, *((px[:]*bins[:])[::-1]))[::-1]):
                    #caution about the axis order :
                    # - xyz for displacements, voxel size and binning size,
                    # - but zyx for derivation axes
                    gradient[t,...,i,j] = g
            print('\r%d (%.01f %%)'%(t, 100*(t+1)/flow.shape[0]), sep=' ', end=' ', flush=True)
        print("\ndone!")
