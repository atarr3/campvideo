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
    
# transcription via google cloud
def transcribe(vid_path,context=None):
    # instantiate storage client and video client
    storage_client = storage.Client('adclass-1286')
    video_client = vi.VideoIntelligenceServiceClient()
    
    # get video files bucket
    bucket = storage_client.get_bucket('video_files')
    
    # upload file to bucket
    blob = bucket.blob(splitext(basename(vid_path))[0])
    blob.upload_from_filename(vid_path)
    
    # speech context
    if context is not None:
        sc = vi.types.SpeechContext(phrases=context)
    
        # transcription configuration
        stc = vi.types.SpeechTranscriptionConfig(
                                        language_code='en-US',
                                        enable_automatic_punctuation=False,
                                        speech_contexts=[sc]
                                                )
    else:
        # transcription configuration
        stc = vi.types.SpeechTranscriptionConfig(
                                        language_code='en-US',
                                        enable_automatic_punctuation=False
                                                )
    
    video_context = vi.types.VideoContext(speech_transcription_config=stc)
    
    # construct video intelligence client request
    video_gcs = 'gs://video_files/' + blob.name
    features = [vi.enums.Feature.SPEECH_TRANSCRIPTION]
    operation = video_client.annotate_video(video_gcs, 
                                            features=features,
                                            video_context=video_context,
                                           )
        
    # get results and delete object from bucket
    try:
        results = operation.result(timeout=600)
    except KeyboardInterrupt:
        blob.delete()
    finally:
        blob.delete()
    
    # print out transcription
    annotation_results = results.annotation_results[0]
    speech_transcription = annotation_results.speech_transcriptions[0]
    alternative = speech_transcription.alternatives[0]
    
    return alternative.transcript
    

def _parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir', 
                        help='Path to video file directory for transcription')
    parser.add_argument('-wf','--wmp_file',default=None,
                        help='Path to WMP data file for matching videos in the '
                             'WMP file to those in vid_dir')
    parser.add_argument('-nf','--names_file',default=None,
                        help='Path to names data file for building speech context'
                             ' from names')

    return parser.parse_args(argv)
    
if __name__ == '__main__':
    # get CL arguments
    args = _parse_arguments(sys.argv[1:])
    vid_dir,wmp,names = args.vid_dir,args.wmp_file,args.names_file
    
    # get video paths
    vid_paths = [join(root,fname) for root,fold,fnames in os.walk(vid_dir) 
                 for fname in fnames if fname.endswith('.mp4')]               
    # filter down to matched videos
    if wmp is not None:
        matched = _matched_vids(vid_paths,wmp)
    else:
        matched = vid_paths
    nvids = len(matched)
    
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