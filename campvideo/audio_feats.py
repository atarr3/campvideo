from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

from argparse import ArgumentParser
from campvideo.audio import Audio
from os.path import join,basename,splitext

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('vid_dir',help='Path to directory containing .mp4 '
                        'files for feature extraction')
    parser.add_argument('-fs','--feature-set',choices=['all','best','no-joint'],
                        default='best',help='Feature set to use when computing '
                        'the audio feature')
    parser.add_argument('-mf','--matches-file',default=None,help='Optional path ' 
                        'to file for filtering out unmatched files in vid_dir.'
                        ' File must contain a column \'uid\' containing the '
                        'list of filenames to keep without the extension')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()
    
    vid_dir = args.vid_dir
    matches_path = args.matches_file
    feature_set = args.feature_set
    
    # get video paths
    fpaths = [join(root,fname) for root,_,fnames in os.walk(vid_dir)
              for fname in fnames if fname.endswith(('.mp4','.wav','.wmv'))]
    # sort
    fpaths = sorted(fpaths,key=lambda s: basename(s))
    
    # filter out unmatched files if wmp_path specified
    if matches_path is not None:
        # video IDs
        matches = pd.read_csv(matches_path,usecols=['uid']
                             ).drop_duplicates(subset='uid'
                             ).set_index('uid')
        fpaths_filtered = [fpath for fpath in fpaths 
                           if splitext(basename(fpath))[0] in matches.index]
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
    feat = pd.DataFrame(index=matches.index,columns=list(range(fdim)),dtype=float)
        
    # compute feature
    for fpath in fpaths_filtered:
        # file ID
        fname = splitext(basename(fpath))[0]
        # feature
        aud = Audio(fpath,fs=22050,nfft=1024,pre_emph=0.95)
        # add to dataframe
        feat.loc[fname] = aud.audiofeat(feature_set=feature_set)
        
    # save in vid_dir
    feat.to_csv(join(vid_dir,'mfeats_' + feature_set + '.csv'))

if __name__ == '__main__':
    main()