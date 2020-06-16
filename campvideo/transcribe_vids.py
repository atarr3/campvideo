import argparse
import os

from campvideo import Video
from os.path import basename,exists,join,splitext
from timeit import default_timer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir',metavar='vid-dir',
                        help='Path to video file directory for transcription')
    parser.add_argument('-up','--use-punct',action='store_true',default=False,
                        help='Enables punctuation annotation for the transcript')

    return parser.parse_args()

def main():
    # get CL arguments
    args = parse_arguments()
    vid_dir = args.vid_dir
    use_punct = args.use_punct

    # get video paths
    fpaths = [join(root,fname) for root,fold,fnames in os.walk(vid_dir)
                                   for fname in fnames
                                       if fname.endswith('.mp4')]
    n_vids = len(fpaths)

    # output directory for transcripts (in root of vid_dir)
    if not exists(join(vid_dir,'transcripts')):
        os.mkdir(join(vid_dir,'transcripts'))

    # debug file
    with open(join(vid_dir,'transcription_log.txt'),'w') as lf:    
        # transcribe
        for i,fpath in enumerate(fpaths):
            print('Transcribing video %d of %d... ' % (i+1,n_vids),end='',flush=True)
            s = default_timer()
            
            # video name
            cur_name = splitext(basename(fpath))[0]
            # transcript filename
            tpath = join(vid_dir,'transcripts',cur_name + '.txt')
            
            # check if video already transcribed
            if exists(tpath):
                print('Transcription already exists')
                continue
    
            # transcribe video
            v = Video(fpath)
            try:
                cur_trans = v.transcribe(use_punct=use_punct)
            except Exception as e:
                msg = 'Failed on video `%s` with error: `%s`' % (fpath,str(e))
                print(msg,file=lf)
                print('Failed')
                continue
            with open(tpath,'wb') as tf:
                    tf.write(cur_trans.encode('utf-8'))
            print('Done in %4.1f seconds!' % (default_timer()-s))

if __name__ == '__main__':
    main()
