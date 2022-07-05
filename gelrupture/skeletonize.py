import sys, os.path, argparse
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
from skimage import morphology
import read_lif as lif
import check
import h5py



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeletonize networks from LIF file.')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('-s','--series', type=int, help='The series to analyse')
    parser.add_argument('-o','--output',default='skeleton', type=str, help = 'output path and prefix')
    parser.add_argument('--preblur',type=float, default=1., help='Amount of gaussian preblur')
    parser.add_argument('--tmax', type =int, help = 'The time lapse till which we do networkization analysis')
    parser.add_argument('--thr', default=1., type=float, help='At how many time the mean of the filtered image the threshold should be set.')
    parser.add_argument('--no_erosion', action='store_true', help='Whether to erode the binary image before skeletonization.')
    parser.add_argument('--bottom', default=0, type=int, help='Where to cut the bottom.')
    parser.add_argument('--top', type=int, help='Where to cut the top.')
    #parser.add_argument('--cut', type=int, default=0, help='Margin to cut on the XY sides of the image')
    parser.add_argument('--no_clean_git', action='store_true', help='Bypass checking wether my git repository is clean (Use only for tests).')
    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)
    if not args.no_clean_git:
        args.commit = check.clean_git(__file__).hexsha
    args.datetime = datetime.now().isoformat()
    rea = lif.Reader(args.filename, False)
    if not hasattr(args, 'series') or args.series is None:
        args.series = rea.chooseSerieIndex()
    ser = rea.getSeries()[args.series]
    if not hasattr(args, 'tmax') or args.tmax is None:
        args.tmax = ser.getNbFrames()
    if not hasattr(args, 'top') or args.top is None:
        args.top = ser.getFrameShape()[0]
    assert args.bottom < args.top

    #prepare output
    with h5py.File(args.output+".h5", "a") as h5file:
        #save pixel size
        px = h5file.require_dataset("voxel_size_in_microns", data= np.array([ser.getVoxelSize(d+1)*1e6 for d in range(3)]), shape=(3,), dtype=np.float64)
        sk = h5file.require_dataset(
            "binary_skeleton",
            (args.tmax, args.top - args.bottom, ser.getFrameShape()[1], ser.getFrameShape()[2]),
            dtype=np.bool,
            chunks=True,
            compression="gzip")
        #save all analysis parameters
        for k,v in args.__dict__.items():
            sk.attrs[k] = v
        #perform actual computation
        for t in range(args.tmax):
            #load only planes from bottom to top
            #binarize plane by plane
            fr_bi = np.zeros(sk.shape[1:], dtype=bool)
            for i,z in enumerate(range(args.bottom, args.top)):
                plane = ser.get2DSlice(T=t, Z=z)
                fr_bi[i] = gaussian_filter(plane.astype(float), args.preblur) > args.thr * plane[plane<255].mean()
            #remove shot noise by eroding
            if not args.no_erosion:
                fr_bi = binary_erosion(fr_bi)
            #actual skeletonization. takes time.
            sk[t] = morphology.skeletonize_3d(fr_bi)
            print('\r%d (%.01f %%)'%(t, 100*t/args.tmax), sep=' ', end=' ', flush=True)
        print("\ndone for %s!"%ser.getName())

        # #position of squelton pixels
        # pos = np.vstack(np.where(fr_sk_cut>0)).T
        # #link neighbouring pixels
        # maxbondlength = 1.8
        # tree = KDTree(pos, 12)
        # bonds = tree.query_pairs(maxbondlength, output_type='ndarray')
        # #save positions and bonds
        # exportname = "{}_squeleton_t{:03d}".format(args.output, t)
        # np.savetxt(exportname+".xy", pos[:,::-1] * px)
        # np.savetxt(exportname+".bonds", bonds, fmt='%d')
