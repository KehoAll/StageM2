import os, os.path
import argparse
import numpy as np
import read_lif as lif

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export z intensity profiles of all series in a LIF file')
    parser.add_argument('filename', type= str, help = 'path and name of the LIF file')
    parser.add_argument('output', type=str, help = 'output path and file name')
    parser.add_argument('-s','--series', type=int, help='The first series to analyse. (the only one if endseries is None.)')
    args = parser.parse_args()
    rea = lif.Reader(args.filename,True)
    print(f'{len(rea.getSeries())} Series')
    ser = lif.SerieCollection(rea.getSeries()[args.series:])
    profiles = np.zeros((ser.getNbFrames(), ser.getFrameShape()[0]))
    for t, fr in ser.enumByFrame():
        profiles[t] = fr.mean(axis=(1,2))
        print(
            '\r%d (%.01f %%)'%(t+1, 100*t/profiles.shape[0]), 
            sep=' ', end=' ', flush=True
        )
    np.save(args.output, profiles)
    print('done!')
