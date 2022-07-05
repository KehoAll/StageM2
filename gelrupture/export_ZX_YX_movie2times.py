import os, os.path, time
import argparse
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter
try:
    import cupy as cp
    from cupyx.scipy import ndimage as ndi
    xp = cp
    asnumpy = cp.asnumpy
    device_name = "gpu"
except:
    from scipy import ndimage as ndi
    xp = np
    asnumpy = np.asarray
    device_name = "cpu"
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import read_lif as lif
#import check

def normalize_stack(stack, px, preblur=1., profile=None):
    if profile is None:
        profile = stack.mean((1,2))
    f0 = ndi.gaussian_filter(stack.astype(xp.float32), preblur*px[-1]/px[:])
    f0 /= profile[:,None,None]
    return f0/f0.max()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a video of orthogonal views')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('Z', type=int, help='Plane at which to show XY plane')
    parser.add_argument('Y', type=int, help='Plane at which to show XZ plane')
    parser.add_argument('X', type=int, help='Plane at which to show YZ plane')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    parser.add_argument('--endseries', type=int, help='The last series to analyse (if a time aquitision was split in more than one series).')
    parser.add_argument('--start_time', type=int, default=0, help='The first time to display')
    parser.add_argument('--length', default=None, type=int, help='The number of frames to display')
    parser.add_argument('--Zfactor',type=float, default=1., help='Multiplicative factor on Z distances, e.g. due to refractive index corrections.')
    parser.add_argument('--Zmin', default=None, type=int, help='First plane to normalize in intensity. Useful when the bottom wall is not flat.')
    parser.add_argument('--Zmax', default=None, type=int, help='Last plane to normalize in intensity. Useful when the top wall is not flat.')
    parser.add_argument('-o','--output',default='orthogonal', type=str, help = 'output path and prefix')
    parser.add_argument('--imageSequence',default=False, type=bool, help = 'whether to output as a sequence of png images. Default is a mp4 movie.')
    #parser.add_argument('--no_clean_git',type=bool, default=False, help='Bypass checking wether my git repository is clean (Use only for tests).')

    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)
    # if not args.no_clean_git:
    #     args.commit = check.clean_git(__file__).hexsha
    args.datetime = datetime.now().isoformat()
    rea = lif.Reader(args.filename, quick=True)
    if not hasattr(args, 'series') or args.series is None:
        args.series = rea.chooseSerieIndex()
    if not hasattr(args, 'endseries') or args.endseries is None:
        args.endseries = args.series

    ser = lif.SerieCollection(rea.getSeries()[args.series:args.endseries+1])
    nbFrames = ser.getNbFrames()
    shape = ser.getFrameShape()
    if args.length is None:
        args.length = nbFrames - args.start_time -1

    if args.Zmin is None:
        args.Zmin = 0
    if args.Zmax is None:
        args.Zmax = shape[0]

    #normalization profile computed at start_time
    fr = xp.array(ser.getFrame(args.start_time))
    profile = fr.mean((1,2))
    #saturates profile to avoid bright noise in dark planes
    profile[:args.Zmin] = profile[args.Zmin]
    profile[args.Zmax:] = profile[args.Zmax-1]

    #pixel size in ZYX order
    px = np.asarray([ser.getVoxelSize(d+1)*1e6 for d in range(3)])[::-1]
    px[0] *= args.Zfactor

    #set formatting options
    #cmap = 'gray'
    lc = (1,1,1,0.2)
    fontprops = fm.FontProperties(size=12)

    fig, axs = plt.subplots(
        2,2, figsize=(6,6), sharex='col', sharey='row',
        subplot_kw={'aspect': 'equal'},
        gridspec_kw={'width_ratios':[shape[2]*px[2],shape[0]*px[0]], 'height_ratios':[shape[0]*px[0],shape[1]*px[1]]}
    )

    to_update = {}
    stack = normalize_stack(fr, px, preblur=0, profile=profile)
    contrast = 1/xp.percentile(stack[args.Z], 95)
    stack = xp.minimum(1, contrast*stack)
    newstack = xp.minimum(1, contrast * normalize_stack(xp.array(ser.getFrame(args.start_time+1)), px, preblur=0, profile=profile))
    #ZX slice
    to_update['ZXslice'] = axs[0,0].imshow(
        np.dstack([
            asnumpy(fr[:,args.Y-5:args.Y+6,:].mean(1))
            for fr in [stack, newstack, stack]
        ]),
        extent=(
            0, shape[2]*px[2],
            shape[0]*px[0],0,
        ),
    )
    axs[0,0].invert_yaxis()
    axs[0,0].set_ylabel('Z (µm)')
    axs[0,0].axhline(args.Z*px[0], color=lc)
    axs[0,0].axvline(args.X*px[2], color=lc)

    #YX slice
    to_update['YXslice'] = axs[1,0].imshow(
        np.dstack([
            asnumpy(fr[args.Z])
            for fr in [stack, newstack, stack]
        ]),
        extent=(
            0, shape[2]*px[2],
            0, shape[1]*px[1]
        ),
    )
    axs[1,0].axhline(args.Y*px[1], color=lc)
    axs[1,0].axvline(args.X*px[2], color=lc)
    axs[1,0].set_xlabel('X (µm)')
    axs[1,0].set_ylabel('Y (µm)')

    #ZY slice
    to_update['ZYslice'] = axs[1,1].imshow(
        np.dstack([
            asnumpy(fr[:,:,args.X-5:args.X+6].mean(2).T)
            for fr in [stack, newstack, stack]
        ]),
        extent=(
            0, shape[0]*px[0],
            0, shape[1]*px[1],
        ),
    )
    axs[1,1].axhline(args.Y*px[1], color=lc)
    axs[1,1].axvline(args.Z*px[0], color=lc)
    axs[1,1].set_xlabel('Z (µm)')

    axs[0,1].set_axis_off()

    #annotations
    scalebar = AnchoredSizeBar(axs[0,0].transData, 50, '50 µm', 'lower left', pad=0.2, color='white',frameon=False,size_vertical=1,fontproperties=fontprops)
    axs[1,0].add_artist(scalebar)

    plt.tight_layout()

    #axs[i].axis('off')
    to_update['timer'] = axs[0,1].text(
        0.5,0.5, f't = {time.strftime("%H:%M:%S", time.gmtime(0))}',
        fontsize =12, weight='bold',
        horizontalalignment='center', verticalalignment='center',
        transform=axs[0,1].transAxes
    )

    def update_graph(t):
        newstack = xp.minimum(1, contrast * normalize_stack(xp.array(ser.getFrame(t+1)), px, preblur=0, profile=profile))
        to_update['ZXslice'].set_data(np.dstack([
            asnumpy(fr[:,args.Y-5:args.Y+6,:].mean(1))
            for fr in [stack, newstack, stack]
        ]))
        to_update['YXslice'].set_data(np.dstack([
            asnumpy(fr[args.Z])
            for fr in [stack, newstack, stack]
        ]))
        to_update['ZYslice'].set_data(np.dstack([
            asnumpy(fr[:,:,args.X-5:args.X+6].mean(2).T)
            for fr in [stack, newstack, stack]
        ]))
        to_update['timer'].set_text(f't = {time.strftime("%H:%M:%S", time.gmtime((t-args.start_time)*ser.getTimeLapse()))}')
        stack[:] = newstack

    ani = FuncAnimation(fig, update_graph, frames=range(args.start_time, args.start_time+args.length))

    if args.imageSequence:
        ani.save(
            args.output+'.png', writer="imagemagick",
            dpi=150,
            metadata = vars(args),
            progress_callback = lambda t, nbFrames: print(f'\r{t}/{nbFrames}', end='', flush=True)
        )
    else:
        ani.save(
            args.output+'.mp4',
            dpi=150, fps=25,
            metadata = vars(args),
            progress_callback = lambda t, nbFrames: print(f'\r{t}/{nbFrames}', end='', flush=True)
        )
    print()

    # for t, fr in ser.enumByFrame():
    #     print(f'\r{t}/{nbFrames}', end='', flush=True)
    #     if t<args.start_time or t-args.start_time>=args.length:
    #         continue
    #     stack = normalize_stack(fr, px, preblur=0, profile=profile)
    #     #ZX slice
    #     axs[0,0].imshow(
    #         stack[:,args.Y-5:args.Y+6,:].mean(1),
    #         cmap = 'hot', norm=color_norm,
    #         extent=(
    #             0, shape[2]*px[2],
    #             shape[0]*px[0],0,
    #         ),
    #         #aspect = 'equal'
    #     )
    #     axs[0,0].invert_yaxis()
    #     axs[0,0].set_ylabel('Z')
    #     axs[0,0].axhline(args.Z*px[0], color=lc)
    #     axs[0,0].axvline(args.X*px[2], color=lc)
    #
    #     #YX slice
    #     axs[1,0].imshow(
    #         stack[args.Z],
    #         cmap = 'hot', norm=color_norm,
    #         extent=(
    #             0, shape[2]*px[2],
    #             0, shape[1]*px[1]
    #         ),
    #         #aspect = 'equal'
    #     )
    #     axs[1,0].axhline(args.Y*px[1], color=lc)
    #     axs[1,0].axvline(args.X*px[2], color=lc)
    #     axs[1,0].set_xlabel('X')
    #     axs[1,0].set_ylabel('Y')
    #
    #     #ZY slice
    #     axs[1,1].imshow(
    #         stack[:,:,args.X-5:args.X+6].mean(2).T,
    #         cmap = 'hot', norm=color_norm,
    #         extent=(
    #             0, shape[0]*px[0],
    #             shape[1]*px[1],0,
    #         ),
    #     )
    #     axs[1,1].axhline(args.Y*px[1], color=lc)
    #     axs[1,1].axvline(args.Z*px[0], color=lc)
    #     axs[1,1].set_xlabel('Z')
    #
    #     axs[0,1].set_axis_off()
    #
    #     #annotations
    #     scalebar = AnchoredSizeBar(axs[0,0].transData, 50, '50 µm', 'lower left', pad=0.2, color='white',frameon=False,size_vertical=1,fontproperties=fontprops)
    #     axs[1,0].add_artist(scalebar)
    #     #axs[i].axis('off')
    #     axs[0,1].text(
    #         0.5,0.5, f't = {timedelta(seconds=((t-args.start_time)*ser.getTimeLapse()))}',
    #         fontsize =12, weight='bold',
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=axs[0,1].transAxes
    #     )
    #
    #     plt.tight_layout()
    #     plt.show()
    #     break




#r = lif.Reader('/media/mleocmach/Leocmach_1/Akash/20211116/20211116_1_CAS2_GDL2.5_AC20x .lif', quick=False)

# fig, axs = plt.subplots(
#     2,2, figsize=(10,10), sharex='col', sharey='row',
#     subplot_kw={'aspect': 'equal'},
#     gridspec_kw={'width_ratios':[ims[0].shape[1],ims[0].shape[0]], 'height_ratios':[ims[0].shape[0],ims[0].shape[1]]}
# )
# Z = 22
# Y = 51
# X = 63
# scale = 1
# lc = (1,1,1,0.2)
# axs[1,0].imshow(np.dstack([im[Z] for im in ims])[...,[0,1,0]], 'gray')
# axs[1,0].axhline(Y, color=lc)
# axs[1,0].axvline(X, color=lc)
# inslice = np.abs(pts[:,0]-Z)<1
# axs[1,0].quiver(pts[inslice,2], pts[inslice,1], d[inslice,2], -d[inslice,1],color='y', units='xy', scale=scale)
# axs[1,0].quiver(pts[inslice,2], pts[inslice,1], displ[inslice,2], -displ[inslice,1],color='b', units='xy', scale=scale)
# axs[1,0].set_xlabel('X')
# axs[1,0].set_ylabel('Y')
#
# axs[0,0].imshow(np.dstack([im[:,Y] for im in ims])[...,[0,1,0]], 'gray')
# axs[0,0].axhline(Z, color=lc)
# axs[0,0].axvline(X, color=lc)
# inslice = np.abs(pts[:,1]-Y)<1
# axs[0,0].quiver(pts[inslice,2], pts[inslice,0], d[inslice,2], d[inslice,0],color="y", units='xy', scale=scale)
# axs[0,0].quiver(pts[inslice,2], pts[inslice,0], displ[inslice,2], displ[inslice,0],color="b", units='xy', scale=scale)
# axs[0,0].set_ylim(axs[0,0].get_ylim()[::-1])
# axs[0,0].set_ylabel('Z')
#
# axs[1,1].imshow(np.dstack([im[:,:,X].T for im in ims])[...,[0,1,0]], 'gray')
# axs[1,1].axhline(Y, color=lc)
# axs[1,1].axvline(Z, color=lc)
# inslice = np.abs(pts[:,2]-X)<1
# axs[1,1].quiver(pts[inslice,0], pts[inslice,1], d[inslice,0], -d[inslice,1],color='y', units='xy', scale=scale)
# axs[1,1].quiver(pts[inslice,0], pts[inslice,1], displ[inslice,0], -displ[inslice,1],color='b', units='xy', scale=scale)
# axs[1,1].set_xlabel('Z')
#
# axs[0,1].set_visible(False)
#
# axs[1,0].set_xlim(40,100)
# axs[1,0].set_ylim(80,20)
# axs[0,0].set_ylim(10,40)
# axs[1,1].set_xlim(10,40)
# plt.tight_layout()
#
# yi=250
# yf = 260
# serie_start  = 8
# ti = 58 #flast fully stationary frame
# ts  =ser.getTimeLapse()
# serie_break = 13
# tbreak = 70
# #serie = [4,5,7,12,12,13,14]
# #TF = [57,21,30,30,70,30,70]
# serie = [8,8,8,9,10]
# TF = [57,59,75,30,10]
# zcalibration = 1.55
# fig,axs = plt.subplots(5,1, figsize = (5,9.), sharex = 'col')
# for i,s in enumerate(serie):
#     ser = r.getSeries()[s]
#     frame = ser.getFrame(T=TF[i])
#     intmax = frame[::-1,yi:yf,:].mean(1).max()
#     print(intmax)
#     axs[i].imshow(frame[::-1,yi:yf,:].mean(1), cmap = 'hot', vmin=0, vmax=intmax, extent=(0,frame.shape[1]*1e6*ser.getVoxelSize(2),
#                                                                                       0,frame.shape[0]*1e6*ser.getVoxelSize(3)*zcalibration),aspect = 'equal')
#     fontprops = fm.FontProperties(size=12)
#     scalebar = AnchoredSizeBar(axs[i].transData, 50, '50 µm', 'lower left', pad=0.2, color='white',frameon=False,size_vertical=1,fontproperties=fontprops)
#     axs[i].add_artist(scalebar)
#     axs[i].axis('off')
#     axs[i].text(160,115, 't = %d s'%((s-8)*Nbframes*+ts+ TF[i]*ts - ti*ts),fontsize =12,color = 'w',weight='bold')
#
#     #axs[i].text(50,115, 't = %d'%((s-4)*78+ TF[i]),fontsize =8,color = 'w')
# for ax in axs:
#     ax.invert_xaxis()
# plt.tight_layout()
# #plt.savefig('/media/asingh/Akash01/20211116/movie/20211116_creep_Y=255_snapshot.pdf')
