import os, os.path
import argparse
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_opening
from scipy.spatial import cKDTree as KDTree
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu, threshold_li
from texture.grid import RegularGrid
from texture.texture import bin_texture, bin_changes
import read_lif as lif
import check
import json
import h5py

def normalize_stack(stack, px, preblur=1., profile=None):
    if profile is None:
        profile = np.ma.array(stack, mask=stack>254).mean((1,2))
    profile = np.where(1+profile**2>1, profile, 1)
    f0 = gaussian_filter(stack.astype(float), preblur*px[0]/px[::-1])
    f0 /= profile[:,None,None]
    return (f0/f0.max()*255).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform discreet texture analysis between planes of successive stacks of a LIF serie')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    parser.add_argument('--endseries', type=int, help='The last series to analyse (if a time aquitision was split in more than one series).')
    parser.add_argument('-o','--output',default='dirdip', type=str, help = 'output path and prefix. If the output.h5 file contains a "displYX" dataset, uses it as a guess of the global shear.')
    parser.add_argument('--preblur',type=float, default=1., help='Amount of gaussian preblur in XY')
    #parser.add_argument('--Zpreblur',type=float, default=0., help='Amount of gaussian preblur in Z')
    parser.add_argument('--Zfactor',type=float, default=1., help='Multiplicative factor on Z distances, e.g. due to refractive index corrections.')
    parser.add_argument('--radius',type=int, default=8, help='The PIV window is 2*radius+1 in X and Y')
    parser.add_argument('--zradius',type=int, default=4, help='The PIV window is 2*zradius+1 in Z')
    parser.add_argument('--minZ',type=int, help='The lowest plane to consider, must be at least zradius (Default zradius).')
    parser.add_argument('--maxZ',type=int, help='The highest plane to consider, at least zradius below the last plane (Default -zradius).')
    parser.add_argument('--gridstep', type=float, help='The step of the grid in X and Y (Default to radius).')
    parser.add_argument('--gridzstep', type=float, help='The step of the grid in Z (Default to zradius).')
    parser.add_argument('--no_clean_git',type=bool, default=False, help='Bypass checking wether my git repository is clean (Use only for tests).')

    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)
    if not args.no_clean_git:
        args.commit = check.clean_git(__file__).hexsha
    args.datetime = datetime.now().isoformat()
    rea = lif.Reader(args.filename, False)
    if not hasattr(args, 'series') or args.series is None:
        args.series = rea.chooseSerieIndex()
    if not hasattr(args, 'endseries') or args.endseries is None:
        args.endseries = args.series

    sers = rea.getSeries()[args.series:args.endseries+1]
    nbFrames = sum(ser.getNbFrames() for ser in sers)
    ser = sers[0]
    if not hasattr(args, 'minZ') or args.minZ is None or args.minZ<args.zradius:
        args.minZ = args.zradius
    if not hasattr(args, 'maxZ') or args.maxZ is None or args.maxZ+args.zradius>ser.getFrameShape()[0]:
        args.maxZ = ser.getFrameShape()[0] - args.zradius -1

    if not hasattr(args, 'gridstep') or args.gridstep is None:
        args.gridstep = args.radius
    if not hasattr(args, 'gridzstep') or args.gridzstep is None:
        args.gridzstep = args.zradius

    #voxel size in XYZ order (microns)
    px = np.asarray([ser.getVoxelSize(d+1)*1e6 for d in range(3)])
    px[-1] *= args.Zfactor

    #shape of the region where points of interest can be
    shape = np.array(ser.getFrameShape()) - 2*args.radius
    shape[0] = args.maxZ - args.minZ

    #Prepare a BC lattice of possible points of interest
    period = 4
    latticecell = np.vstack((
        (np.arange(period)==period-1)[None,:,None] & (np.arange(period)==period-1)[None,None,:],
        (np.arange(period)==period//2-1)[None,:,None] & (np.arange(period)==period//2-1)[None,None,:]
    ))
    lattice3D = np.tile(latticecell, shape//[2, period, period])
    #all the bond lengths in a cell
    lattice_lengths = np.linalg.norm(px * [[period,0,0], [period, period, 0], [0,0,2], [period//2, period//2,1]], axis=-1)
    #Define the cutoff distance in microns
    radius_um = 0.5 * (lattice_lengths.max() + 2*lattice_lengths.min())

    #Prepare the grid on which the texture will be binned
    #The grid is in microns
    grid = RegularGrid(
        [args.minZ,args.radius,args.radius]*px[::-1],
        [args.gridzstep, args.gridstep, args.gridstep]*px[::-1],
        shape//[args.gridzstep, args.gridstep, args.gridstep]
        )


    #prepare output
    with h5py.File(args.output+".h5", "a") as h5file:
        #save pixel size
        h5file.require_dataset(
            "voxel_size_in_microns",
            data= px,
            shape=(3,), dtype=np.float64)
        #prepare timestamps array on disk
        timestamps = h5file.require_dataset(
            "timestamps",
            (nbFrames, ser.getFrameShape()[0]),
            dtype=float
            )
        #prepare the output arrays on disk
        grp = h5file.create_group("texture_piv")
        #the number of matrices binned in each grid element
        Ntot = grp.require_dataset(
            "Ntot",
            (nbFrames-1,) + grid.shape,
            dtype='int64')
        #the sum of the texture matrices on each grid element.
        sumM = grp.require_dataset(
            "sumM",
            (nbFrames-1,) + grid.shape + (6,),
            dtype='float64')
        #the sum of the C matrices on each grid element (geometrical changes).
        sumC = grp.require_dataset(
            "sumC",
            (nbFrames-1,) + grid.shape + (3,3),
            dtype='float64')
        #the sum of the T matrices on each grid element (topological changes).
        sumT = grp.require_dataset(
            "sumT",
            (nbFrames-1,) + grid.shape + (6,),
            dtype='float64')
        #the number of appearing matrices binned in each grid element
        Na_tot = grp.require_dataset(
            "Na_tot",
            (nbFrames-1,) + grid.shape,
            dtype='int64')
        #the number of disappearing matrices binned in each grid element
        Nd_tot = grp.require_dataset(
            "Nd_tot",
            (nbFrames-1,) + grid.shape,
            dtype='int64')


        #check the existence of an array holding the global shear
        displYX = h5file.get('displYX', default=np.zeros((nbFrames-1, ser.getFrameShape()[0], 2), np.int64))
        #knock it
        displYX = np.rint(displYX[:]).astype(np.int64)
        #save all analysis parameters
        for k,v in args.__dict__.items():
            grp.attrs[k] = v

        #allocate memory
        # flowxy = np.zeros(ser.getFrameShape()+[2])
        # flowxz = np.zeros_like(flowxy)
        # flowyz = np.zeros_like(flowxy)
        # weights = np.zeros(ser.getFrameShape()+[3])
        #perform actual computation
        ti = 0
        for ser in sers:
            print("\n%s"%(ser.getName()))
            #save timestamps
            ts = ser.getTimeStamps()
            nts = len(ts)//ser.getFrameShape()[0]
            timestamps[ti:ti+nts] = ts[:nts*ser.getFrameShape()[0]].reshape((nts, ser.getFrameShape()[0]))
            for t, fr in ser.enumByFrame():
                #condition the current stack.
                f1 = normalize_stack(fr, px, args.preblur)

                if t+ti > 0:
                    #select points of interest in bright zones of the image at t0
                    threshold = np.array([threshold_li(plane) for plane in f0])
                    thresholded = binary_opening(f0>threshold[:,None,None])
                    pts = np.column_stack(np.where(
                        lattice3D & thresholded[args.minZ:args.maxZ, args.radius:-args.radius, args.radius:-args.radius]
                        ))
                    pts += [args.minZ, args.radius, args.radius]
                    #filter out points advected out by global shear
                    pts1 = pts[:,1:] + displYX[t+ti-1, pts[:,0]]
                    good = np.all(pts1>=args.radius, axis=-1) & np.all(pts1+args.radius<ser.getFrameShape()[1:], axis=-1)
                    pts = pts[good]

                    #displacement by phase cross correlation of the points of interest
                    displ = np.zeros(pts.shape)
                    for u,(i,j,k) in enumerate(pts):
                        dj, dk = displYX[t+ti-1, i]
                        displ[u] = phase_cross_correlation(
                            #advect the windows by global shear
                            f1[i-args.zradius:i+args.zradius+1, j+dj-args.radius:j+dj+args.radius+1, k+dk-args.radius:k+dk+args.radius+1],
                            f0[i-args.zradius:i+args.zradius+1, j-args.radius:j+args.radius+1, k-args.radius:k+args.radius+1],
                            upsample_factor=10, return_error=False,
                            ) + [0,dj,dk] #add back the global shear
                    #the two sets of points to use are the points found at t0
                    #and the same points advected by the displacements
                    points = [pts, pts+displ]
                    #points are bounded if they are closer than the cutoff
                    bonds = [
                        np.array([[a,b]
                        for a,b in KDTree(points*px[::-1]).query_pairs(radius_um)])
                        for points in points
                        ]
                    #static texture at t0
                    sumM[t+ti-1], Ntot[t+ti-1] = bin_texture(
                        points[0]*px[::-1], bonds[0],
                        grid, points_per_bond=2
                        )
                    #bin changes between the two frames
                    sumC[t+ti-1], Nc_tot, sumT[t+ti-1], Na_tot[t+ti-1], Nd_tot[t+ti-1] = bin_changes(
                        points[0]*px[::-1], points[1]*px[::-1],
                        bonds[0], bonds[1],
                        grid, points_per_bond=2
                        )
                f0 = np.copy(f1)
                print('\r%d (%.01f %%)'%(t, 100*(t+ti+1)/sumM.shape[0]), sep=' ', end=' ', flush=True)
            ti += ser.getNbFrames()
        print("\ndone!")
