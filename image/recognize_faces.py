import argparse
import cv2
import numpy as np
import os
import sys

import face

from timeit import default_timer

# get keyframes
def get_keyframes(vpaths,kf_inds):
    kf = []
    for i,vpath in enumerate(vpaths):
        cap = cv2.VideoCapture(vpath)
        inds = kf_inds[i]
        for ind in inds:
            cap.set(1,ind)
            ret,cf = cap.read()
            kf.append(cf)
            
    return kf

# generator for reading keyframe text files
def docgen(fpaths):
    for fpath in fpaths:
        with open(fpath,'r') as fn:
            yield [int(kf) for kf in fn.read().strip(', ').split(', ')
                    if len(kf) != 0]
            
def main(args):
    summ_dir = args.summ_dir
    vid_dir = args.vid_dir
    baseline_im = args.baseline_im
    
    # path to keyframe index files
    kf_paths = [os.path.join(root,name) for root,dirs,files in os.walk(summ_dir)
                                        for name in files 
                                        if name.endswith('.txt')]
    # path to video files
    vid_paths = [os.path.join(root,name) for root,dirs,files in os.walk(vid_dir)
                                         for name in files 
                                         if name.endswith('.mp4')]
    # video identifiers
    ids = [kf_path.split(os.path.sep)[-2] for kf_path in kf_paths]
    nvids = len(ids)
    # keyframe indices
    kf_inds = [ind for ind in docgen(kf_paths)]
    # number of keyframes per video
    kf_lengths = [len(kf_ind) for kf_ind in kf_inds]
    
    # instantiate face recognition object
    recognizer = face.Recognition()
    # add baseline image as identity
    recognizer.add_identity(cv2.imread(baseline_im),threshold=0.94)
    
    # batching
    batch_size = 250
    cp = []
    cum_sums = np.cumsum(kf_lengths)
    for i,cum_sum in enumerate(cum_sums):
        if cum_sum > batch_size:
            cp.append(i)
            cum_sums -= cum_sums[i-1]
    # add beginning/end cutpoint if not already in
    if 0 not in cp: cp.insert(0,0)  
    if nvids not in cp: cp.append(nvids)
    nbatches = len(cp) - 1
    
    # compute identies
    vid_results = np.zeros(nvids,dtype=bool)
    
    s = default_timer()
    
    for i in range(nbatches):
        # video indices
        curvid_ind = slice(cp[i],cp[i+1])
        print("Processing videos {0} through {1} of {2}".format(cp[i]+1,cp[i+1],nvids))
        # keyframe indices for each video
        curkf_ind = kf_inds[curvid_ind]
        curkf_lengths = [len(item) for item in curkf_ind]
        # get batch of images
        curbatch = get_keyframes(vid_paths[curvid_ind],curkf_ind)
        im_results = recognizer.identify(curbatch)
        # mark true if any image in video contains identity
        vid_results[curvid_ind] = [np.any(item) for item in np.split(im_results,np.cumsum(curkf_lengths)[:-1])]
        
    print("Time elapsed: {0} hours".format((default_timer()-s)/3600))
    
    # write to output file
    out_file = os.path.join(summ_dir,'results.txt')
    with open(out_file,'w') as fn:
        for i,result in enumerate(vid_results):
            fn.write('{0},{1}\n'.format(ids[i], result))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()  
    parser.add_argument('summ_dir',type=str,help='Directory of video summaries')
    parser.add_argument('vid_dir',type=str,help='Directory of video files')
    parser.add_argument('baseline_im',type=str,help='Path to baseline image for identification')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))