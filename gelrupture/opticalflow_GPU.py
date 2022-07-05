import os, os.path
import argparse
from datetime import datetime
import numpy as np
#try:
import cupy as cp
from cupyx.scipy import ndimage as ndi
from cucim.skimage.util import img_as_float32
from cucim.skimage.registration import optical_flow_tvl1
xp = cp
asnumpy = cp.asnumpy
device_name = "gpu"
# except:
#     from scipy import ndimage as ndi
#     from skimage.util import img_as_float32
#     from skimage.registration import optical_flow_tvl1
#     xp = np
#     asnumpy = np.asarray
#     device_name = "cpu"
import read_lif as lif
import check
import json
import h5py
import logging, time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform 2D optical flow between planes of successive stacks of a LIF serie')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    parser.add_argument('--endseries', type=int, help='The last series to analyse (if a time aquitision was split in more than one series).')
    parser.add_argument('-o','--output',default='flowGPU', type=str, help = 'output path and prefix')
    parser.add_argument('--compression', default=False, help='Compression algorithm. Default is False. Possible values are "gzip" or "lzf"')
    parser.add_argument('--Zmin', type=int, default=0, help='Lowest plane to process')
    parser.add_argument('--Zmax', type=int, default=None, help='First plane not to process')
    #parser.add_argument('--preblur',type=float, default=1., help='Amount of gaussian preblur in XY')
    #parser.add_argument('--Zpreblur',type=float, default=0., help='Amount of gaussian preblur in Z')
    #parser.add_argument('--pyrsteps',type=int, default=4, help='Number of steps in the pyramid used by Farneback optical flow algorithm')
    #parser.add_argument('--square_farneback',type=bool, default=False, help='Optical flow algorithm uses square filtering instead of isotropic Gaussian. Causes anisotropy problem but is marginally faster.')
    #parser.add_argument('--binning',type=int, default=8, help='Binning of the output flow field in X and Y. Chose a power of 2.')
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

    logging.basicConfig(filename=args.output+".log", encoding='utf-8', level=logging.DEBUG)
    print(f'device: {device_name}')

    ser = lif.SerieCollection(rea.getSeries()[args.series:args.endseries+1])
    nbFrames = ser.getNbFrames()

    if args.Zmax is None:
        args.Zmax = ser.getFrameShape()[0]

    #useful spatial shape
    shape = (args.Zmax-args.Zmin, )+ tuple(ser.getFrameShape()[1:])

    #prepare output
    with h5py.File(args.output+".h5", "a") as h5file:
        #prepare the output array on disk
        flow = h5file.require_dataset(
            "FlowField3D",
            (nbFrames-1,)+shape+(3,),
            dtype='float32',
            compression=args.compression#"lzf"#, compression_opts=9
            )
        #prepare timestamps array on disk
        timestamps = h5file.require_dataset(
            "timestamps",
            (nbFrames, ser.getFrameShape()[0]),
            dtype=float
            )
        #save all analysis parameters
        for k,v in args.__dict__.items():
            flow.attrs[k] = v

        #save timestamps
        ts = ser.getTimeStamps()
        timestamps[:len(ts)] = ts

        #allocate memory
        f0 = xp.zeros(shape, xp.float32)
        f1 = xp.zeros_like(f0)
        wvu = np.zeros((3,)+shape, np.float32)

        #perform actual computation
        for t, fr in ser.enumByFrame():
            logging.info(f'Load t={t:03d} to GPU...')
            #load to GPU
            f1[:] = img_as_float32(xp.asarray(fr[args.Zmin:args.Zmax]))
            logging.info('done!')
            if t > 0:
                logging.info(f"Compute optical flow on {device_name}...")
                wvu_GPU = optical_flow_tvl1(f0, f1)
                logging.info('done!')
                time.sleep(0.1)
                logging.info(f"Transfer to RAM...")
                time.sleep(0.1)
                wvu[:] = asnumpy(wvu_GPU)
                logging.info('done!')
                time.sleep(0.1)
                logging.info(f"Write to disk...")
                time.sleep(0.1)
                flow[t-1] = np.moveaxis(wvu, 0, -1)
                logging.info('done!')
                time.sleep(0.1)
            logging.info(f"Swap frames...")
            f0, f1 = f1, f0
            logging.info('done!')
            print('\r%d (%.01f %%)'%(t, 100*t/flow.shape[0]), sep=' ', end=' ', flush=True)
