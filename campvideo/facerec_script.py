import argparse
import cv2
import os
import pickle

from campvideo import Keyframes
from face_recognition import face_encodings, face_locations
from os.path import basename,dirname,join,splitext
from timeit import default_timer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir',type=str,
                        help="""Directory of videos in .mp4 or .wav format. Directory
                        structure must be in the form:
                        path\to\vid_dir\<elect>\<state>_<candidate>\<filename>""")
    parser.add_argument('summ_dir',type=str,
                        help="""Directory of keyframe indices in .txt format""")

    return parser.parse_args()

def main():
    # get arguments
    args = parse_arguments()
    vid_dir, summ_dir = args.vid_dir, args.summ_dir

    # get list of videos and summaries
    fpaths_vid = [join(root,fname) for root,_,fnames in os.walk(vid_dir)
                                       for fname in fnames
                                           if fname.endswith(('.mp4','.wav'))]
    fpaths_vid = sorted(fpaths_vid,key=lambda x: basename(x).lower())

    fpaths_summ = [join(root,fname) for root,_,fnames in os.walk(summ_dir)
                                        for fname in fnames
                                            if fname.endswith('.txt')]
    fpaths_summ = sorted(fpaths_summ,key=lambda x: basename(dirname(x)).lower())

    # get face encodings for videos
    out = []
    n = len(fpaths_vid)
    for (i,(fpath_vid,fpath_sum)) in enumerate(zip(fpaths_vid,fpaths_summ)):
        print("Processing video %d of %d.... " % (i+1,n),end='',flush=True)
        s = default_timer()

        # construct Keyframes object
        with open(fpath_sum,'r') as fh:
            kf_ind = [int(x) for x in fh.read().split(',')]
        kf = Keyframes.fromvid(fpath_vid,kf_ind)
        
        # get encodings for detected faces
        encs = [enc for im in kf.ims 
                        for enc in face_encodings(
                                           cv2.cvtColor(im,cv2.COLOR_BGR2RGB),
                                           face_locations(
                                               cv2.cvtColor(im,cv2.COLOR_BGR2RGB),
                                               model='cnn'),
                                           model="large",
                                           num_jitters=100
                                           )]

        # get candidate information from the video [elec,state_cand,fname]
        metadata = fpath_vid.split(os.path.sep)[-3:]
        # convert to (elec\state\cand,uid) format
        metadata = (join(metadata[0],*metadata[1].split('_')),
                    splitext(metadata[-1])[0])

        # store results
        out.append((metadata,encs))
        print("Done in %.1fs" % (default_timer()-s))

    # save in video directory
    pickle.dump(out,open(join(vid_dir,'face_encodings.pkl'),'wb'))

if __name__ == '__main__':
    main()
