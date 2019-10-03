from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

from argparse import ArgumentParser
from campvideo.audio import Spectrogram
from os.path import join,basename,splitext

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('vid_dir',help='Path to directory containing .mp4 '
                        'files for feature extraction')
    parser.add_argument('-fs','--feature_set',choices=['all','best','no_joint'],
                        default='best',help='Which feature set to use when '
                        'computing the audio feature')
    parser.add_argument('-wp','--wmp_path',default=None,help='Optional path to '
                        'WMP file for filtering out unmatched files in vid_dir')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()
    
    vid_dir = args.vid_dir
    wmp_path = args.wmp_path
    feature_set = args.feature_set
    
    # get video paths
    fpaths = [join(root,fname) for root,_,fnames in os.walk(vid_dir)
              for fname in fnames if fname.endswith(('.mp4','.wav'))]
    # sort
    fpaths = sorted(fpaths,key=lambda s: splitext(basename(s))[0])
    
    # filter out unmatched files if wmp_path specified
    if wmp_path is not None:
        # video IDs
        wmp = pd.read_csv(wmp_path,usecols=['uid']
                         ).drop_duplicates(subset='uid'
                         ).set_index('uid')
        fpaths_filtered = [fpath for fpath in fpaths 
                           if splitext(basename(fpath))[0] in wmp.index]
    else:
        fpaths_filtered = fpaths
        
    # feature dimension
    if feature_set == 'best':
        fdim = 452
    elif feature_set == 'all':
        fdim = 636
    else:
        fdim = 412
        
    # feature data frame
    feat = pd.DataFrame(index=wmp.index,columns=list(range(fdim)),dtype=float)
        
    # compute feature
    for fpath in fpaths_filtered:
        # file ID
        fname = splitext(basename(fpath))[0]
        # feature
        s = Spectrogram(fpath,fs=22050,nfft=1024,pre_emph=0.95)
        # add to dataframe
        feat.loc[fname] = s.audiofeat(feature_set=feature_set)
        
    # save in vid_dir
    wmp.to_csv(join(vid_dir,'mfeats_' + feature_set + '.csv'))

if __name__ == '__main__':
    main()