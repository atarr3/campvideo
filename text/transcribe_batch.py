import argparse
import os
import pandas as pd

from google.cloud import videointelligence_v1beta2 as vi
from google.cloud import storage

CMAG_DIR  = 'E:\\Users\\Alex\\OneDrive\\Documents\\Research\\campvideo\\U2D Documents'
NAMES_DIR = 'E:\\Users\\Alex\\Desktop'
OUT_DIR   = 'E:\\Users\\Alex\\Desktop'

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

# returns list of videos in given directory which have a matching entry in WMP
def matched_vids(vid_paths):
    # load files
    cmag = pd.read_csv(os.path.join(CMAG_DIR,'issuedetectiontable020418.csv'),
                       usecols=['uid']).drop_duplicates(subset='uid')
    ids = list(cmag.uid)
    fpaths = [fpath for fpath in vid_paths if os.path.splitext(os.path.basename(fpath))[0] in ids]
    
    return fpaths
    
def get_metadata(fpath):
    # get election/year and state/name
    elye,stna = fpath.split('\\')[-3:-1]
    # convert to names.csv-friendly form
    year = elye[3:]
    elect = ELECTS[elye[0:3]]
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
                       
            
# transcription via google cloud
def transcribe(vid_path,context):
    # instantiate storage client and video client
    storage_client = storage.Client('adclass-1286')
    video_client = vi.VideoIntelligenceServiceClient()
    
    # get video files bucket
    bucket = storage_client.get_bucket('video_files')
    
    # upload file to bucket
    blob = bucket.blob(os.path.splitext(os.path.basename(vid_path))[0])
    blob.upload_from_filename(vid_path)
    
    # speech context
    sc = vi.types.SpeechContext(phrases=context)
    
    # transcription configuration (need to add context for names)
    stc = vi.types.SpeechTranscriptionConfig(language_code='en-US',
                                             speech_contexts=[sc])
    video_context = vi.types.VideoContext(speech_transcription_config=stc)
    
    # construct video intelligence client request
    video_gcs = 'gs://video_files/' + blob.name
    features = [vi.enums.Feature.SPEECH_TRANSCRIPTION]
    operation = video_client.annotate_video(video_gcs, 
                                            features=features,
                                            video_context=video_context,
                                            location_id='us-east1')
        
    # get results and delete object from bucket
    results = operation.result(timeout=600)
    blob.delete()
    
    # print out transcription
    annotation_results = results.annotation_results[0]
    speech_transcription = annotation_results.speech_transcriptions[0]
    alternative = speech_transcription.alternatives[0]
    
    return alternative.transcript
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir', 
                        help='Path to video file directory for transcription')
    args = parser.parse_args()
    # get arguments
    vid_dir = args.vid_dir    
    # get video paths
    vid_paths = [os.path.join(root,fname) for root,fold,fnames in os.walk(vid_dir) 
                 for fname in fnames if fname.endswith('.mp4')]               
    # filter down to matched videos
    matched = matched_vids(vid_paths)
    nvids = len(matched)
    # open names file
    names = pd.read_csv(os.path.join(NAMES_DIR,'names.csv'),keep_default_na=False,encoding='mbcs')
    
    # # read data frame
    # cmag = pd.read_csv(os.path.join(CMAG_DIR,'new.csv'),
    #                    encoding='mbcs')
    #                    
    # # insert empty column after 'script' column for new captions
    # if 'google_caption' not in cmag.columns:
    #     cmag.insert(cmag.columns.get_loc('caption')+1,'google_caption',None)    
    
    # output directory
    if not os.path.exists(os.path.join(OUT_DIR,'transcripts')):
        os.mkdir(os.path.join(OUT_DIR,'transcripts'))
    
    # transcribe
    for i,vid_path in enumerate(matched):
        print('Transcribing video {0} of {1}...'.format(i+1,nvids),end=' ',flush=True)
        # video id
        cur_id = os.path.splitext(os.path.basename(vid_path))[0]
        # metadata
        elect,year,state,cand = get_metadata(vid_path)
        # check if video already transcribed
        if os.path.exists(os.path.join(OUT_DIR,'transcripts',cur_id+'.txt')):
            print('transcription already exists')
            continue
            
        # get context
        sub = names[(names.election == elect) & (names.year == int(year)) & 
                    (names.state == state) & ((names.D.str.contains(cand)) | 
                    (names.R.str.contains(cand)) | (names['T'].str.contains(cand)))]
        try:
            context = build_context(sub)
        except IndexError:
            print('entry not found for {0} in the {1} {2} election in {3}'.format(cand,year,elect,state))
            break
        try:
            cur_trans = transcribe(vid_path,context)
            # cmag.set_value(cmag.index[cmag.uid == cur_id],'google_caption',cur_trans)
            with open(os.path.join(OUT_DIR,'transcripts',cur_id+'.txt'),'w') as tf:
                tf.write(cur_trans)
            print('success!')
        except:
            print('failed')
            continue
            
    # save file
    # cmag.to_csv(os.path.join(CMAG_DIR,'new.csv'),index=False)