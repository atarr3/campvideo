import argparse
import os
import stat

from campvideo.video import Video
from cv2 import imwrite
from os.path import normpath,join,relpath,splitext,basename,dirname
from shutil import rmtree
from timeit import default_timer

# command line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()  
    parser.add_argument('vid_dir',type=str,help='Directory of videos in .mp4 or .wav format')
    parser.add_argument('-l1',default=1.5,type=float,
                        help='Penalty for uniqueness. Default value is 1.5')
    parser.add_argument('-l2',default=3.5,type=float,
                        help='Penalty for summary length. Default value is 3.5')
    parser.add_argument('-n',default=50,type=int,
                        help='Number of iterations to run the optimization algorithm. Default value is 50')
    parser.add_argument('-wf','--write-frames',action='store_true',default=False,
                        help='Write keyframes to .png in output directory')

    return parser.parse_args()

# script for summarizing a collection of videos
def main():
    args = parse_arguments()
    
    # video directory and video paths
    vid_dir = args.vid_dir
    l1,l2 = args.l1, args.l2
    wf = args.write_frames
    
    # list of filepaths
    fpaths = [join(root,name) for root,dirs,files in os.walk(vid_dir)
                                  for name in files 
                                      if name.endswith(('.mp4','.wmv'))]
    n = len(fpaths)
                                         
    # output directory for summary (mimics folder structure of input)
    summ_dir = normpath(vid_dir) + '_summaries'
    # delete directory if it exists
    if os.path.exists(summ_dir):
        os.chmod(summ_dir, stat.S_IWUSR) # grant all privileges
        rmtree(summ_dir,ignore_errors=True)
    
    os.mkdir(summ_dir)
    
    for i,fpath in enumerate(fpaths):
        print('Processing video {0} of {1}... '.format(i+1,n),end='',flush=True)
        s = default_timer()          
        
        # instantiate video stream object for original video
        v = Video(fpath)
        
        # compute keyframes
        kf_ind = v.summarize(l1=l1,l2=l2)
                
        # output directory for summary
        out_dir = join(summ_dir,relpath(dirname(fpath),vid_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # save keyframes indices
        kf_file = join(out_dir,splitext(basename(fpath))[0] + '.txt')
        kf_out = []
        with open(kf_file,'w') as fn:
            for i,kf in enumerate(v.frames(kf_ind)):
                kf_out.append(str(kf_ind[i]))
                # save keyframe image
                if wf:
                    name = splitext(basename(fpath))[0]
                    fname = join(out_dir,name+'_{0:04d}.png'.format(kf_ind[i]))
                    imwrite(fname,kf)
            # write frames to `keyframes.txt`
            fn.write(','.join(kf_out))
        
        # video summarized
        f = default_timer() - s
        print("Done in %.1fs" % f)
        
if __name__ == '__main__':
    main()