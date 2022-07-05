
import openpiv.tools
import openpiv.process
import openpiv.scaling
import openpiv.validation
import numpy as np
import openpiv.filters
from colloids import lif
from matplotlib import pyplot as plt
from colloids import lif
from scipy.ndimage.filters import gaussian_filter
import pandas as pd

def PIV_RAW(frame_a, frame_b, window_size, overlap, search_area_size, dt):
    # appliies PIV for two arrays of shear images. dt is an unknown paramter which is used to give the approximate displacement vectors (dispx,dispy) from the velocity field vector(u,v) since we don't know the time between the two images
    #convert the frame_a, and frame_b which are read by the openpiv.tools.imread to np.int32 format using astype
	u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size =window_size  , overlap=overlap, dt=dt, search_area_size=search_area_size, sig2noise_method='peak2peak')

	dispx = u * dt
	dispy = v* dt

	sigmaW = sig2noise.sum()
    
	WdispX = (dispx*sig2noise).sum()/sigmaW #weighted average displacement in x
	WdispY = (dispy*sig2noise).sum()/sigmaW #weighted average displacement in y

	netdispX= dispx - WdispX
	netdispY = dispy -WdispY

	netU = netdispX/dt
	netV = netdispY/dt

	x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size= window_size, overlap=overlap)
    
	piv_raw =  [u,v,netU,netV,WdispX, WdispY, x, y, sig2noise]    	
	return piv_raw

def scalevalues(im, m=None, M=None):
    if m is None:
        m = im.min()
    if M is None:
        M = im.max()
    return (im-m)/(M-m)

def PIV_Z(fr,fr1,z,window_size, overlap, search_area_size,threshold, dir, shear = '0-->10', image = True, background = True,dt=0.001):
    # applies PIV between 2 sets of shear data at different z plane
    u=[]
    v = []
    netU= []
    netV = []
    WdispX = []
    WdispY = []
    x = []
    y = []
    plane_height=[]
    sig2noise  = []
    for n in range(10,z,5):
        filter_a = gaussian_filter(fr[n].astype(float), 101)
        filter_b = gaussian_filter(fr1[n].astype(float), 101)
        frame_a = fr[n]- filter_a
        frame_b = fr1[n]- filter_b
        u1, v1, netU1, netV1, WdispX1, WdispY1, x1, y1, sig2noise1 = PIV_RAW(frame_a, frame_b, window_size=window_size, overlap=overlap, search_area_size=search_area_size ,dt=dt)
        netU1, netV1, mask = openpiv.validation.sig2noise_val( netU1, netV1, sig2noise1, threshold)
        netU1, netV1 = openpiv.filters.replace_outliers( netU1, netV1, method='localmean', max_iter=10, kernel_size=2)
        if image == True:
            if background == True:
                P= plt.figure(figsize=(10,10))
                plt.imshow(np.dstack((scalevalues(frame_a, 0,100), scalevalues(frame_b,0,100), np.zeros((1024,1024)))), origin='lower')
                plt.text(700, 800, shear+',z=%s'%n,fontsize=15, color='w', bbox=dict(facecolor='white', alpha=0.5))
                plt.quiver(x1,y1,u1,v1,color='b', scale = 50000, scale_units='inches')
            else:
                P= plt.figure(figsize=(10,10))
                plt.text(700, 800, shear,fontsize=15, color='w', bbox=dict(facecolor='white', alpha=0.5))
                plt.quiver(x1,y1,u1,v1,color='b', scale = 50000, scale_units='inches')
            plt.savefig(dir+'z%s.png'%n)
        u.append(u1)
        v.append(v1)
        netU.append(netU1)
        netV.append(netV1)
        WdispX.append(WdispX1)
        WdispY.append(WdispY1)
        sig2noise.append(sig2noise1)
        plane_height.append(n)
        x.append(x)
        y.append(y)
        
    stack = [u,v,netU,netV,WdispX, WdispY, x, y, sig2noise, plane_height]
    return stack
    
if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply PIV between two stack of images at different shear')
    parser.add_argument('lif_dir', help='Location of the lif image file')
    parser.add_argument('series1', type=int, help= 'The first stack')
    parser.add_argument('series2', type=int, help= 'The 2nd stack')
    parser.add_argument('z', type=int, help='The height of the stack to pe processsed')
    parser.add_argument('dir', help='The location where the results are stored')
    parser.add_argument('shear', help='The applied shear')
    parser.add_argument('image',help='Boolean to specify if plot of the PIV should be made or not')
    parser.add_argument('background',help='Boolean to specify if the quiver qre to be superimposed on the overlapped image')
    parser.add_argument('--ws','--window_size',default=96, type=int, help = 'size  of the PIV window')
    parser.add_argument('--ol','--overlap',default=48, type=int,  help='number of pixel that can be overlapped between 2 windows')
    parser.add_argument('--sas','--search_area_size',type=int, default=192, help=' The area in which the window is allowed for image correlation')
    parser.add_argument('--threshold', default=1.1, type=float, help='The threshold signal to noise ratio below which all the displacement vectors are neglected and replaced')
    
    args=parser.parse_args()
    if args.image:
        image= True
    else:
        image= False
    if args.background:
        background= True
    else:
        background= False
    r = lif.Reader(args.lif_dir)
    s= r.getSeries()[args.series1]
    fr= s.getFrame()
    s1 = r.getSeries()[args.series2]
    fr1 = s1.getFrame()
    u,v,netU,netV,WdispX, WdispY, x, y, sig2noise, plane_height= PIV_Z(fr,fr1,z=args.z,window_size=args.ws, overlap=args.ol, search_area_size=args.sas,threshold=args.threshold, dir=args.dir, shear = args.shear, image = image, background = background)
    data = {'WdispX':WdispX, 'WdispY':WdispY, 'plane_height':plane_height}
    df = pd.DataFrame(data, columns = ['WdispX', 'WdispY', 'plane_height'])
    df.to_csv(args.dir+'shear'+ args.shear +'.csv')
    
    
