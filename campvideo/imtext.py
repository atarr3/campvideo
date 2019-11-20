from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from campvideo import Keyframes
from os.path import join,splitext,basename,dirname,exists
from timeit import default_timer

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
    
    # check if directories found
    if not exists(summ_dir):
        raise FileNotFoundError('Summary directory `%s` not found' % summ_dir)
    if not exists(vid_dir):
        raise FileNotFoundError('Video directory `%s` not found' % vid_dir)
    
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
    
    # error logging
    log_path = join(summ_dir,'log.txt')
    # remove existing log file from previous runs
    if exists(log_path): os.remove(log_path)
    
    # iterate through videos
    n = len(fpaths_vid) 
    for i,(fpath_summ, fpath_vid) in enumerate(zip(fpaths_summ,fpaths_vid)):
        # progress
        print('Processing video %4d of %4d... ' % (i+1,n),end='',flush=True)
        
        # skip to next video if already processed
        if os.path.exists(join(dirname(fpath_summ),'image_text.txt')):
            print('Already processed')
            continue
        
        # check that video and summary correspond to one another
        if basename(dirname(fpath_summ)) != splitext(basename(fpath_vid))[0]:
            print()
            raise Exception("Summary and video structure not equivalent between directories")
        
        # get keyframe indices as list
        with open(fpath_summ,'r') as fh:
            kf_ind = [int(num) for num in fh.read().strip(',').split(',')]
		
		
        s = default_timer()
        with open(log_path,'a') as fh:
            print('Processing video %d of %d... ' % (i+1,n),file=fh)	          
			# get frames from video and create Keyframes object
            print('\tReading keyframes... ',file=fh,end='',flush=True)
            try:
                kf = Keyframes.fromvid(fpath_vid,kf_ind)
            except KeyboardInterrupt:
                msg = 'Process terminated on video `%s`' % fpath_vid
                print(msg,file=fh)
                print('Failed')
                continue
            except Exception as e:
                msg = 'Failed on video `%s` with error: `%s`' % (fpath_vid,str(e))
                print(msg,file=fh)
                print("Failed.")
                continue
            print('Done!',file=fh)
			
			# get image text for each image
            print('\tDetecting text... ',file=fh,end='',flush=True)
            try:
                texts = kf.keyframes_text()
                # remove empty lists
                texts = [item for item in texts if item != []]
            except Exception as e:
                # error logging in summ_dir
                msg = 'Failed on video `%s` with error: `%s`' % (fpath_vid,str(e))
                print(msg,file=fh)
                print("Failed.")
                continue
            print('Done!',file=fh)
			
			# write to same directory as summary
            print('\tSaving results... ',file=fh,end='',flush=True)
            with open(join(dirname(fpath_summ),'image_text.txt'),'wb') as out:
                # tab-delimited words, newline for each keyframe
                out.write('\n'.join(['\t'.join(text) for text in texts]
                  ).encode('utf-8'))
            print("Done!\n",file=fh)
            
        print('Done in %4.1f seconds!' % (default_timer()-s))
    
    print('\nAll files processed! Log file saved to %s' % log_path)
                  
if __name__ == '__main__':
    main()