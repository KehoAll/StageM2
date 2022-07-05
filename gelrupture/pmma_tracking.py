import numpy as np
import matplotlib.pyplot as plt
from colloids import lif
from colloids.track import CrockerGrierFinder
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from scipy.ndimage.filters import gaussian_filter
from pims import Bioformats, pipeline
import trackpy as tp
from mayavi import mlab


def track3d(path,series, diameter,separation,smoothing_size, output,minmass=100,noise_size=1.5):
    frames = Bioformats(path, series = series)
    if frames.ndim==3:
        frames.bundle_axes = 'zyx'
        im  = frames[0]
        features = tp.locate(raw_image=im, diameter=diameter, minmass=minmass, separation=separation, noise_size=noise_size,smoothing_size=smoothing_size)
        features.to_csv(output)
    
    elif frames.ndim == 4:
        frames.bundle_axes = 'zyx'
        frames.iter_axes = 't'
        features= tp.batch(frames = frames, diameter=diameter, minmass=minmass, separation=separation, noise_size=noise_size,smoothing_size=smoothing_size)
        features.to_csv(output)
        
def track2d(path,series, diameter,separation,output,noise_size=1.5, minmass=100):
    frames = Bioformats(path, series = series)
    if frames.ndim==2:
        frames.bundle_axes = 'yx'
        im  = frames[0]
        features = tp.locate(raw_image=im, diameter=diameter, minmass=minmass, separation=separation, noise_size=noise_size)
        features.to_csv(output)
    
    elif frames.ndim == 3:
        frames.bundle_axes = 'yx'
        frames.iter_axes = 't'
        features= tp.batch(frames = frames, diameter=diameter, minmass=minmass, separation=separation, noise_size=noise_size)
        features.to_csv(output)
 
 
def Mayavi3d(features, path,series,axes, fmin, fmax, tf):
    frames = Bioformats(path, series = series)
    f = pd.read_csv(features)
    f1 = f.loc[:,'z':]
    if frames.ndim==3:
        frames.bundle_axes = 'zyx'
        im  =frames[0]
        imT = im.T
        if axes =='y':
            fs = f1[(f1.y>=fmin) & (f1.y<=fmax)]
            (x,y,z,mass) = (fs.x,fs.y,fs.z,fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='y_axes', colormap = 'hot')
            mlab.show()
        elif axes =='z':
            fs = f1[(f1.z>=fmin) & (f1.z<=fmax)]
            x,y,z,mass = (fs.x,fs.y, fs.z, fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='z_axes', colormap = 'hot')
            mlab.show()
        elif axes =='x':
            fs = f1[(f1.x>=fmin) & (f1.x<=fmax)]
            x,y,z,mass = (fs.x,fs.y, fs.z, fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='x_axes', colormap = 'hot')
            mlab.show()
    if frames.ndim==4:
        frames.bundle_axes = 'zyx'
        frames.iter_axes = 't'
        im = frames[tf]
        imT = im.T
        if axes =='y':
            fs = f1[(f1.y>=fmin) & (f1.y<=fmax)& (f1.frame== tf)]
            (x,y,z,mass) = (fs.x,fs.y,fs.z,fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='y_axes', colormap = 'hot')
            mlab.show()
        elif axes =='z':
            fs = f1[(f1.z>=fmin) & (f1.z<=fmax)&(f1.frame== tf)]
            x,y,z,mass = (fs.x,fs.y, fs.z, fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='z_axes', colormap = 'hot')
            mlab.show()
        elif axes =='x':
            fs = f1[(f1.x>=fmin) & (f1.x<=fmax)&(f1.frame== tf)]
            x,y,z,mass = (fs.x,fs.y, fs.z, fs.mass)
            mlab.points3d(x,y,z, mass, colormap='viridis', scale_mode = 'none',scale_factor = 10)
            mlab.volume_slice(imT, plane_orientation='x_axes', colormap = 'hot')
            mlab.show()    

              
