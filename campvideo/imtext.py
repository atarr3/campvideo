from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from campvideo import Image
from campvideo import Video
from os.path import join,splitext,basename,dirname

# command line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()  
    parser.add_argument('summ_dir',type=str,help='Directory of video summaries, each saved in `keyframes.txt`')
    parser.add_argument('vid_dir',type=str,help='Directory of videos in .mp4 or .wav format')

    return parser.parse_args()

# script for summarizing a collection of videos
def main():
    args = parse_arguments()
    
    # summary directory and keyframe filepaths
    summ_dir, vid_dir = args.summ_dir, args.vid_dir 
    fpaths_summ = [join(root,name) for root,dirs,files in os.walk(summ_dir)
                                       for name in files 
                                           if name.endswith(('keyframes.txt'))]
    fpaths_vid = [join(root,name) for root,dirs,files in os.walk(vid_dir)
                                      for name in files 
                                          if name.endswith(('.mp4','.wmv'))]
    
    # there must be a corresponding keyframe file for each video file. matches
    # are dictated by the parent folder of `keyframes.txt`, which should be the
    # name of the video file
    fpaths_summ = sorted(fpaths_summ,
                         key = lambda s: basename(dirname(s)).lower())
    fpaths_vid  = sorted(fpaths_vid,
                         key = lambda s: splitext(basename(s))[0].lower())
    
    # iterate through videos
    n = len(fpaths_vid)
    for i,(fpath_summ, fpath_vid) in enumerate(zip(fpaths_summ,fpaths_vid)):
        # progress
        print('Processing video %d of %d... ' % (i+1,n),end='',flush=True)
        # check that video and summary correspond to one another
        if basename(dirname(fpath_summ)) != splitext(basename(fpath_vid))[0]:
            print()
            raise Exception("Summary and video structure not equivalent between directories")
        
        # get keyframe indices as list
        with open(fpath_summ,'r') as fh:
            kf_ind = [int(num) for num in fh.read().strip(',').split(',')]
            
        # get frames from video and create list of Images
        vid = Video(fpath_vid)
        ims = [Image(frame) for frame in vid.frames(kf_ind)]
        
        # get image text for each image
        texts = [im.image_text() for im in ims]
        
        # write to same directory with summary is
        with open(join(dirname(fpath_summ),'image_text.txt'),'wb') as fh:
            # tab-delimited words, newline for each keyframe
            fh.write('\n'.join(['\t'.join(text) for text in texts]).encode('utf-8'))
        
        print('Done!')
            
            
if __name__ == '__main__':
    main()