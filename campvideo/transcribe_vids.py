import argparse
import os
import pandas as pd
import sys

from google.cloud import videointelligence as vi
from google.cloud import storage
from os.path import basename,exists,join,splitext

STATES = {"AL":"Alabama", "AK":"Alaska","AR":"Arkansas","AZ":"Arizona","CA":"California",
          "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida",
          "GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana",
          "IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine",
          "MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota",
          "MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska",
          "NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
          "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
          "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island",
          "SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas",
          "UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
          "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","US":"United States"}

ELECTS = {'sen':'Senate','hou':'House','gov':'Gubernatorial','pre':'Presidential'}

# returns list of videos in given directory which have a matching entry in WMP data
def _matched_vids(vid_paths,wmp_file):
    # load files
    wmp = pd.read_csv(wmp_file,usecols=['uid']
                     ).drop_duplicates(subset='uid')
    ids = set(wmp.uid)

    return [fpath for fpath in vid_paths if splitext(basename(fpath))[0] in ids]

def _get_metadata(fpath):
    # get election/year and state/name
    elye,stna = fpath.split('\\')[-3:-1]
    # convert to names.csv-friendly form
    year = elye[3:]
    elect = ELECTS[elye[:3]]
    state = STATES[stna[:2]]
    # get district
    if elect == 'House':
         state += ' ' + stna[2:4].lstrip('0')
         name = stna[5:]
    else:
        name = stna[3:]

    return elect,year,state,name

def _build_context(df):
    dem = df.iloc[0]['D'].split(',')
    rep = df.iloc[0]['R'].split(',')
    thi = df.iloc[0]['T'].split(',')

    if dem != '' and rep != '':
        context = list(filter(None,dem+rep))
    else:
        # only retain first third-party candidate
        context = list(filter(None,dem+rep+thi[0]))

    return context

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir',
                        help='Path to video file directory for transcription')
    parser.add_argument('-nf','--names-file',default='',
                        help='Path to names data file for providing phrase hints'
                             ' for speech transcription')

    return parser.parse_args()

if __name__ == '__main__':
    # get CL arguments
    args = parse_arguments()
    vid_dir,names = args.vid_dir,args.names_file

    # get video paths
    fpaths = [join(root,fname) for root,fold,fnames in os.walk(vid_dir)
                                   for fname in fnames
                                       if fname.endswith('.mp4')]
    n_vids = len(fpaths)

    # open names file if given
    if names is not None:
        names = pd.read_csv(names,keep_default_na=False,encoding='mbcs')

    # output directory for transcripts (in root of vid_dir)
    if not exists(join(vid_dir,'transcripts')):
        os.mkdir(join(vid_dir,'transcripts'))

    # transcribe
    for i,vid_path in enumerate(matched):
        print('Transcribing video {0} of {1}...'.format(i+1,nvids),end=' ',flush=True)
        # video id
        cur_id = splitext(basename(vid_path))[0]
        # metadata
        elect,year,state,cand = _get_metadata(vid_path)
        # check if video already transcribed
        if exists(join(vid_dir,'transcripts',cur_id+'.txt')):
            print('transcription already exists')
            continue

        # get context
        if names is not None:
            sub = names[(names.election == elect) & (names.year == int(year)) &
                        (names.state == state) & ((names.D.str.contains(cand)) |
                        (names.R.str.contains(cand)) | (names['T'].str.contains(cand)))]
            try:
                context = _build_context(sub)
            except IndexError:
                print('entry not found for {0} in the {1} {2} election in {3}'.format(cand,year,elect,state))
                break
        else:
            context = None
        # transcribe video
        try:
            cur_trans = transcribe(vid_path,context)
            with open(join(vid_dir,'transcripts',cur_id+'.txt'),'w') as tf:
                tf.write(cur_trans)
            print('success!')
        except KeyboardInterrupt:
            raise
        except:
            print('failed')
            continue
