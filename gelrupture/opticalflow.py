import os, os.path
import argparse
from datetime import datetime
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import read_lif as lif
import check
import json
import h5py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform 2D optical flow between planes of successive stacks of a LIF serie')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    parser.add_argument('--endseries', type=int, help='The last series to analyse (if a time aquitision was split in more than one series).')
    parser.add_argument('-o','--output',default='flow', type=str, help = 'output path and prefix')
    parser.add_argument('--preblur',type=float, default=1., help='Amount of gaussian preblur in XY')
    parser.add_argument('--Zpreblur',type=float, default=0., help='Amount of gaussian preblur in Z')
    parser.add_argument('--pyrsteps',type=int, default=4, help='Number of steps in the pyramid used by Farneback optical flow algorithm')
    parser.add_argument('--square_farneback',type=bool, default=False, help='Optical flow algorithm uses square filtering instead of isotropic Gaussian. Causes anisotropy problem but is marginally faster.')
    parser.add_argument('--binning',type=int, default=8, help='Binning of the output flow field in X and Y. Chose a power of 2.')
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

    flags = 0
    if not args.square_farneback:
        flags &= cv.OPTFLOW_FARNEBACK_GAUSSIAN

    #prepare output
    with h5py.File(args.output+".h5", "a") as h5file:
        #prepare the output array on disk
        flow = h5file.require_dataset(
            "FlowField3D",
            (nbFrames-1, ser.getFrameShape()[0], ser.getFrameShape()[1]//args.binning, ser.getFrameShape()[2]//args.binning, 3),
            dtype='float32')
        #prepare timestamps array on disk
        timestamps = h5file.require_dataset(
            "timestamps",
            (nbFrames, ser.getFrameShape()[0]),
            dtype=float
            )
        #save all analysis parameters
        for k,v in args.__dict__.items():
            flow.attrs[k] = v
        #make a list with the binning value to be multiplied with the opticalflow value to obtain obsolute values in pixel
        bins = h5file.require_dataset(
            "binning_size_in_pixels",
            data= np.array([args.binning, args.binning,1]),
            shape=(3,), dtype=np.int32)
        binning_shape = (flow.shape[1], flow.shape[2], args.binning, flow.shape[3], args.binning)
        #allocate memory
        flowxy = np.zeros(ser.getFrameShape()+[2])
        flowxz = np.zeros_like(flowxy)
        flowyz = np.zeros_like(flowxy)
        weights = np.zeros(ser.getFrameShape()+[3])
        #perform actual computation
        ti = 0
        for ser in sers:
            print("\n%s"%(ser.getName()))
            #save timestamps
            ts = ser.getTimeStamps()
            nts = len(ts)//ser.getFrameShape()[0]
            timestamps[ti:ti+nts] = ts[:nts*ser.getFrameShape()[0]].reshape((nts, ser.getFrameShape()[0]))
            for t, fr in ser.enumByFrame():
                #blur only in XY since PSF is already elongated in Z.
                f1 = gaussian_filter(fr.astype(float), [args.Zpreblur,args.preblur,args.preblur])
                #normalize by intensity profile along z. Absolutely crucial, otherwise Farneback algorithm discards low intensity pixels
                intensity_profile = f1.max((1,2))
                intensity_profile[intensity_profile==0] = 1
                f1 = (f1*(255/intensity_profile)[:,None,None]).astype(np.uint8)
                if t+ti > 0:
                    weights[:] = np.finfo(weights.dtype).eps
                    #unfortunately, cv.alcOpticalFlowFarneback does not work directly in 3D
                    #Optical flow in each XY plane give dX and dY, but no information on dZ
                    for i, (im0, im1) in enumerate(zip(f0, f1)):
                        flowxy[i] = cv.calcOpticalFlowFarneback(
                            im0, im1,
                            None, 0.5, args.pyrsteps, 15, 3, 5, 1.2, flags=flags
                            )
                        #estimate the confidence of each pixel using a Shi-Tomasi criterion
                        weights[i,...,0] = cv.cornerMinEigenVal(im0,3)
                    #unshear f1 along X only (here we suppose dY small)
                    zprofile = (flowxy*weights[...,0,None]).sum((1,2)) / weights[...,0,None].sum((1,2))
                    izprofile = zprofile.astype(int)
                    maxshift = izprofile.max(0)
                    minshift = izprofile.min(0)
                    ptpshift = maxshift - minshift
                    shifted = np.zeros((f1.shape[0], f1.shape[1], f1.shape[2]-ptpshift[0]), np.uint8)
                    for z, im in enumerate(f1):
                        shift = izprofile[z,0]-minshift[0]
                        shifted[z] = im[...,shift:shifted.shape[-1]+shift]
                        #shifted[z] = np.pad(im, pad_width=((0,0), (ptpshift[0]-shift[0],shift[0])), mode='constant')
                    #Optical flow in each XZ where a correspondance is possible between the two times.
                    mshift = max(0, minshift[0])
                    sl = slice(mshift, shifted.shape[2]+mshift)
                    for i in range(f0.shape[1]):
                        fl = cv.calcOpticalFlowFarneback(
                            f0[:,i, sl],
                            shifted[:,i,:],
                            None, 0.5, args.pyrsteps, 15, 3, 5, 1.2, flags=flags
                        )
                        #Outside of the conrresponding range, results correspond to XY results
                        flowxz[:,i,:] = fl.mean(axis=(0,1))
                        #inside the range, use the calculated values
                        flowxz[:,i,sl] = fl
                        #estimate the confidence of each pixel using a Shi-Tomasi criterion
                        weights[:,i,sl,1] = cv.cornerMinEigenVal(f0[:,i,sl],3)
                    #Slicing perpendicular to Y creates independent values and thus noise along Y. Blur along that direction
                    gaussian_filter1d(flowxz, 4, axis=1, output=flowxz)
                    #add back the integer part of the X displacement
                    flowxz[...,0] += np.trunc(zprofile[:,None,None,0])

                    #Optical flow in each YZ where a correspondance is possible between the two times.
                    for i in range(shifted.shape[2]):
                        fl = cv.calcOpticalFlowFarneback(
                            f0[:,:,i+mshift],
                            shifted[:,:,i],
                            None, 0.5, args.pyrsteps, 15, 3, 5, 1.2, flags=flags
                        )
                        flowyz[...,i+mshift,:] = fl
                        #estimate the confidence of each pixel using a Shi-Tomasi criterion
                        weights[...,i+mshift,2] = cv.cornerMinEigenVal(f0[:,:,i+mshift],3)
                    #pad YZ results along X to correspond to the shape of XY results shape
                    mflowyz = flowyz[:,:,sl].mean(2)[:,:,None]
                    flowyz[:,:,:mshift] = mflowyz
                    flowyz[:,:,shifted.shape[2]+mshift:] = mflowyz
                    #Slicing perpendicular to X creates independent values and thus noise along X. Blur along that direction
                    gaussian_filter1d(flowyz, 4, axis=2, output=flowxz)
                    #add a small constant weight to prevent division by zero
                    #weights += 1e-7
                    #weighted binning in XY and saving. Here we suppose opicalflow in XY is the ground truth
                    flow[t+ti-1,...,:2] = (flowxy*weights[...,0,None]).reshape(binning_shape+(2,)).sum(axis=(2,4)) / weights[...,0].reshape(binning_shape).sum(axis=(2,4))[...,None]
                    #take the average of the two measurements of dZ, weighted by confidence
                    flow[t+ti-1,...,2] = ((flowxz[...,1]*weights[...,1] + flowyz[...,1]*weights[...,2])).reshape(binning_shape).sum(axis=(2,4)) / ((weights[...,1]+weights[...,2])).reshape(binning_shape).sum(axis=(2,4))
                f0 = np.copy(f1)
                print('\r%d (%.01f %%)'%(t, 100*(t+ti+1)/flow.shape[0]), sep=' ', end=' ', flush=True)
            ti += ser.getNbFrames()
        print("\ndone!")
