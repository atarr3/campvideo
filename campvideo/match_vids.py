from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

from campvideo.audio import Audio,get_dur,vid2wav
from collections import defaultdict
from itertools import chain,combinations
from os.path import join,basename,dirname,splitext
from tempfile import TemporaryDirectory
from warnings import warn

class FingerprintDB(object):
    # builds the database from all audio / video files in file_dir
    def __init__(self,file_dir):
        # parent directory for database
        self.dir = file_dir
        # initialize fingerprint database and song database
        self._fp_db = defaultdict(list) # list of tuples (songID, offset)
        self._song_db = {}              # tuple of fingerprint and file duration

        # get list of files
        fpaths = [join(root,fname)
                  for root,_,fnames in os.walk(file_dir)
                      for fname in fnames
                          if fname.endswith(('.mp4','.wmv','.wav'))
                 ]

        # get fingerpints for each file
        for fpath in fpaths:
            # basename of file
            fname = splitext(basename(fpath))[0]
            # 0.3712s frames at 5kHz sampling rate, spaced by 11.6ms
            fps,_ = Audio(fpath,nfft=2048,wfunc=np.hanning,wlen=1856,
                                overlap=31/32,scaling='spectrum',mode='magnitude'
                               ).fingerprint(reliability=False)

            # update fingerprint database
            for off,fp in enumerate(fps): self._fp_db[fp].append((fname,off))
            # update song database (contains full fingerprint)
            self._song_db[fname] = (fps,get_dur(fpath))

    # find matching file(s) in database for a given file
    def find_match(self,fpath,dur,threshold=0.3,find_all=False):
        # get fingerprint and reliability bits for unmatched file
        fps,rels = Audio(fpath,nfft=2048,wfunc=np.hanning,wlen=1856,
                               overlap=31/32,scaling='spectrum',mode='magnitude'
                              ).fingerprint()

        # number of sub-fingerprints in unmatched file
        fp_size = len(fps)

        # find match(es)
        if find_all: # return all matches
            matched = set()
            for off,(fp,rel) in enumerate(zip(fps,rels)):
                for cand_id,cand_fp in self._fp_gen(fp,off,rel,fp_size,dur,matched):
                    # no need to check other sub-fps of already matched IDs
                    if _ber(fps,cand_fp,32) < threshold:
                        matched.add(cand_id)
            return list(matched)
        else: # return first match found
            # generator for matches. loops through flipped-bit variants of each
            # unknown sub-fingerprint and check if there's a match
            match_gen = (cand_id
                         for off,(fp,rel) in enumerate(zip(fps,rels))
                             for cand_id,cand_fp in self._fp_gen(fp,off,rel,fp_size,dur)
                                 if _ber(fps,cand_fp,32) < threshold
                        )
            try:
                return [next(match_gen)]
            except StopIteration:
                return []

    # aligned candidate fingerprint generator
    def _fp_gen(self,sub_fp,sub_off,sub_rel,fp_size,dur,matched=[]):
        # flip unreliable bits to generate robust list of unmatched
        # sub-fingerprints
        for cur_sub in _flip_bits(sub_fp,sub_rel):
            for cand_id,cand_off in self._fp_db.get(cur_sub,[]):
                # get candidate fp and duration
                # return fingerprint if valid
                cand_fp,cand_dur = self._song_db[cand_id]
                
                # ignore already matched ids or files with different duration
                if cand_id in matched or abs(cand_dur - dur) > 8:
                    continue
                # alignment for candidate fingerprint
                s = cand_off - sub_off
                e = s + fp_size
                if s >= 0 and e <= len(cand_fp):
                    yield (cand_id,cand_fp[s:e])

# computes power set of an iterable, ignoring the empty set
def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

# computes bit error rate between two fingerprints (given as array of ints)
def _ber(fp1,fp2,nbits):
    return sum(np.fromiter((bin(rep).count('1') for rep in fp1 ^ fp2),dtype=int)) / (len(fp1) * nbits)

# generates candidate sub-fingerprints by flipping unreliable bits
def _flip_bits(fp,rel):
    masks = np.fromiter((np.bitwise_or.reduce(1 << np.array(offsets))
             for offsets in _powerset(rel)),dtype=np.uint32)
    flipped = fp ^ masks
    # add original sub-fingerprint to front of array
    return np.insert(flipped,0,fp)

# converts file to .wav and returns the middle segment of specified duration
def _cut_middle(fpath,out_dir=None,dur=3,verbose=True):
    # get duration
    file_dur = get_dur(fpath)
    # check file duration against clipped duration
    if dur > file_dur:
        dur = int(file_dur // 2)
        if verbose:
            warn("Requested clip duration is longer than the file, clipping to "
                 "middle 50% instead",RuntimeWarning)
    # get starting point of middle segment
    start = (file_dur - dur) / 2
    # cut and convert file to .wav
    out = vid2wav(fpath,out_dir=out_dir,start=start,dur=dur)

    # return original duration
    return file_dur,out

# simple clustering function that combines arrays containing common elements
def _grouper(seq):
    result = []
    while len(seq) > 0:
        # split first group from rest of the list, convert to set
        first, *rest = seq
        first = set(first)

        lf = -1
        while len(first) > lf: # terminates when there is no change to set
            lf = len(first)

            rest2 = []
            # find remaining groups intersecting with current group
            for r in rest:
                cur_set = set(r)
                if first.intersection(cur_set):
                    # take union of sets
                    first |= cur_set
                else:
                    rest2.append(r)
            rest = rest2

        result.append(first)
        seq = rest
    return result

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', help='Path(s) to directories for matching. If '
                        'one directory is given, the files are clustered. If '
                        'two directories are given, the first directory is '
                        'matched to the second directory',nargs='+')
    parser.add_argument('-mf','--match-folders',action='store_true',default=False,
                        help='Match bottom-most folders in first and second '
                        'directories')
    parser.add_argument('-fa','--find-all',action='store_true',default=False,
                        help='Find all matches for each unmatched file')
    parser.add_argument('-d','--duration',type=int,default=3,
                        help='Truncates unmatched files to middle section of '
                        'this duration when matching. If the duration is longer'
                        ' than the video, truncates to 50%')
    parser.add_argument('-t','--threshold',type=float,default=0.3,
                        help='Threshold for determining a match between two '
                        'fingerprints. If the BER is below this value, a '
                        'match is declared')

    return parser.parse_args()

# main script for matching videos / audio files
def main():
    # get CL arguments
    args = parse_arguments()
    
    dirs,match_folders,find_all = args.dirs,args.match_folders,args.find_all
    dur,threshold = args.duration,args.threshold

    # assert no more than 2 directories specified
    assert len(dirs) <= 2, "Too many directories specified"

    # two different directories given
    if len(dirs) == 2 and dirs[0] != dirs[1]:
        # unpack
        dir1,dir2 = dirs
    # one directory given, find duplicates by matching directory to itself
    else:
        dir1 = dirs[0]
        dir2 = dir1
        # overwrite command-line input for find_all
        match_folders = False
        find_all = True

    # match folder to folder
    if match_folders:
        # path to bottom folders for both directories, sorted by folder name.
        # bottom here means the directory which contains no folders
        dpaths1 = sorted([root.lower() for root,folders,fnames in os.walk(dir1)
                          if not folders], 
                         key=lambda s: join(basename(dirname(s)),basename(s))
                        )
        dpaths2 = sorted([root.lower() for root,folders,fnames in os.walk(dir2)
                          if not folders], 
                         key=lambda s: join(basename(dirname(s)),basename(s))
                        ) if dir1 != dir2 else dpaths1
        
        # intersection between both directories
        f1 = set(join(basename(dirname(s)),basename(s)) for s in dpaths1)
        f2 = set(join(basename(dirname(s)),basename(s)) for s in dpaths2)
        joint = f1.intersection(f2)
        func = lambda s: join(basename(dirname(s)),basename(s)) in joint
        # filter
        dpaths1 = list(filter(func,dpaths1))
        dpaths2 = list(filter(func,dpaths2))
        
        if len(dpaths1) == 0:
            raise Exception("No overlapping folders between {0} and {1}".format(dir1,dir2))
    else:
        dpaths1 = [dir1]
        dpaths2 = [dir2]

    # number of folders
    nf = len(dpaths1)

    # result file(s) cleanup if running again
    if os.path.exists(join(dir1,'matches.txt')):
        os.remove(join(dir1,'matches.txt'))
    if os.path.exists(join(dir1,'clusters.txt')):
        os.remove(join(dir1,'clusters.txt'))

    # match folder to folder
    for i,(fold1,fold2) in enumerate(zip(dpaths1,dpaths2)):
        print('Matching directory {0} of {1}'.format(i+1,nf))

        # assert folder structure equivalence between directories. folder
        # structure is equivalent so long as the bottom-level folder name is the
        # same in both directories
        if match_folders:
            assert basename(fold1) == basename(fold2),("Folder names not "
            "equivalent between {0} and {1}").format(fold1,fold2)

        # build database for current folder in 2nd directory
        fp_db = FingerprintDB(fold2)        

        with TemporaryDirectory() as temp:
            # processed file paths and original duration (dur,fpath)
            fpaths = [_cut_middle(join(root,fname),out_dir=temp,dur=dur,verbose=False)
                      for root,_,fnames in os.walk(fold1)
                          for fname in fnames
                              if fname.endswith(('.mp4','.wmv','.wav'))
                     ]

            # find matches ([:-9] truncates _proc.wav from filename) for all
            # video files in current folder
            matches = [(basename(fpath)[:-9],
                        fp_db.find_match(fpath,dur,find_all=find_all,threshold=threshold))
                       for dur,fpath in fpaths
                      ]
        # write results in first directory (unmatched)
        with open(join(dir1,'matches.txt'),'a+') as fh:
            fh.write('\n'.join(
                                [fname+'\t' + ','.join(match)
                                 for fname,match in matches]
                              ) + '\n' # append newline between folders
                    )


        # if only one directory passed, cluster the matches
        if len(dirs) == 1:
            # add unmatched fpath to match list to form preliminary clusters
            for fname,match in matches:
                match.insert(0,fname)

            # cluster the list of matches
            clusters = _grouper([match for _,match in matches])

            # write results in dir1 (unmatched directory)
            with open(join(dir1,'clusters.txt'),'a+') as fh:
                fh.write('\n'.join(
                                   [','.join(cluster)
                                    for cluster in clusters]
                                  ) + '\n' # append newline between folders
                        )

if __name__ == '__main__':
    main()