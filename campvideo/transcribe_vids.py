import argparse
import os
import pandas as pd

from campvideo import Video
from os.path import basename,exists,join,sep,splitext
from timeit import default_timer

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

def get_metadata(fpath):
    # get election/year and state/name
    elye,stna = fpath.split(sep)[-3:-1]
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

def build_context(df):
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

def main():
    # get CL arguments
    args = parse_arguments()
    vid_dir,names_file = args.vid_dir,args.names_file

    # get video paths
    fpaths = [join(root,fname) for root,fold,fnames in os.walk(vid_dir)
                                   for fname in fnames
                                       if fname.endswith('.mp4')]
    n_vids = len(fpaths)

    # open names file if given
    if names_file != '':
        names = pd.read_csv(names_file,keep_default_na=False)

    # output directory for transcripts (in root of vid_dir)
    if not exists(join(vid_dir,'transcripts')):
        os.mkdir(join(vid_dir,'transcripts'))

    # transcribe
    for i,fpath in enumerate(fpaths):
        print('Transcribing video %d of %d... ' % (i+1,n_vids),end='',flush=True)
        s = default_timer()
        
        # video name
        cur_name = splitext(basename(fpath))[0]
        # election metadata
        elect,year,state,cand = get_metadata(fpath)
        # transcript filename
        tpath = join(vid_dir,'transcripts',cur_name + '.txt')
        
        # check if video already transcribed
        if exists(tpath):
            print('transcription already exists')
            continue

        # get context
        if names_file != '':
            sub = names[(names.election == elect) & 
                        (names.year == int(year)) &
                        (names.state == state) & 
                        ((names.D.str.contains(cand)) |
                        (names.R.str.contains(cand)) | 
                        (names['T'].str.contains(cand)))]
            try:
                phrases = build_context(sub)
            except IndexError:
                print('entry not found for %s in the %s %s election in %s' % (cand,year,elect,state))
                raise
        else:
            phrases = []
        # transcribe video
        v = Video(fpath)
        try:
            cur_trans = v.transcribe(phrases=phrases)
            with open(tpath,'wb') as tf:
                tf.write(cur_trans.encode('utf-8'))
            print('Done in %4.1f seconds!' % (default_timer()-s))
        except KeyboardInterrupt:
            raise
        except:
            print('Failed')
            continue

if __name__ == '__main__':
    main()
