import os, os.path
import argparse
from datetime import datetime
import numpy as np
from scipy import fft
from skimage.registration import phase_cross_correlation
import read_lif as lif
import check
import json
import h5py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform phase cross correlation plane by plane between consecutive stacks of a LIF serie')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    parser.add_argument('--endseries', type=int, help='The last series to analyse (if a time aquitision was split in more than one series).')
    parser.add_argument('-o','--output',default='flow', type=str, help = 'output path and prefix')
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

    ser = lif.SerieCollection(rea.getSeries()[args.series:args.endseries+1])
    nbFrames = ser.getNbFrames()
    #allocate memory
    f0 = np.zeros(ser.getFrameShape(), dtype=np.complex128)
    f1 = np.zeros_like(f0)

    #prepare output
    with h5py.File(args.output+".h5", "a") as h5file:
        #prepare the output arrays on disk
        #the number of matrices binned in each grid element
        displYX = h5file.require_dataset(
            "displYX",
            (nbFrames-1, ser.getFrameShape()[0], 2),
            dtype='float64')

        #save all analysis parameters
        for k,v in args.__dict__.items():
            displYX.attrs[k] = v

        #perform actual computation
        ti = 0
        for t, fr in ser.enumByFrame():
            #Fourier transform each plane of the current stack.
            f1[:] = fft.fftn(fr, axes=(1,2))

            if t+ti > 0:
                for z in range(f1.shape[0]):
                    displYX[t+ti-1, z] = phase_cross_correlation(
                        f1[z], f0[z],
                        upsample_factor=10,
                        space="fourier",
                        return_error=False,
                        )
                    print(f'\rt={t:d} ({100*(t+1)/displYX.shape[0]:.01f} %) z={z:d} ({100*(z+1)/f1.shape[0]:.01f} %)', sep=' ', end=' ', flush=True)
            f0[:] = f1
        print("\ndone!")
