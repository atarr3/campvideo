import argparse
import boto
import csv
import gcs_oauth2_boto_plugin
import os
import re
import time

import numpy as np
import xml.etree.ElementTree as ET

from google.cloud import speech
from subprocess import call
from textwrap import wrap

# URI scheme for Cloud Storage.
GOOGLE_STORAGE = 'gs'
BUCKET = 'speech_files-2222017'

# Path to diarization jar file
ROOT = os.path.dirname(os.path.realpath(__file__))
DIARIZE = ROOT+'\\diarization\\LIUM_SpkDiarization.jar'

# Application default credentials provided by env variable
# GOOGLE_APPLICATION_CREDENTIALS

# Get speaker segment intervals from xml file
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    start = []
    end = []
    speaker = []
    
    for child in root[1][1]:
        start.append(float(child.get('start')))
        end.append(float(child.get('end')))
        speaker.append(child.get('speaker'))
        
    speaker = np.array(speaker)
    start = np.array(start)
    end = np.array(end)
    # find transition points
    trans = np.hstack((0,np.where(speaker[1:] != speaker[:-1])[0] + 1))
    # start and end times for each segment
    diffs = np.hstack((np.diff(trans),len(end)-trans[-1]))
    begin = np.array([start[i] for i in trans])
    durs = np.array([end[trans[i]] if diffs[i] == 1 else end[trans[i]+diffs[i]-1] 
           for i in range(len(diffs))]) - begin
           
    return np.array([begin,durs])

# Segment audio file using speaker diarization
def diarization(speech_file):
    # output directory
    out_dir = os.path.dirname(speech_file)
    # execute LIUM diarization code (assumes WAV input)
    call(['java','-jar',DIARIZE,'--doCEClustering','--fInputMask='+
          speech_file,'--sOutputMask=%s.xml','--sOutputFormat=seg.xml,UTF-8',
          os.path.join(out_dir,'speech_file')])
    # parse xml file
    segs = parse_xml(os.path.join(out_dir,'speech_file.xml'))
    # split up audio file in ffmpeg
    print speech_file
    for i in range(segs.shape[1]):
        out_file = speech_file[:-4] + '_' + str(i) + '.raw'
        call(['ffmpeg','-y','-loglevel','quiet','-i',speech_file,'-vn','-ac','1',
              '-ar','16k','-f','s16le','-acodec','pcm_s16le','-ss',str(segs[0,i]),
              '-t',str(segs[1,i]),out_file])
    
    # delete xml file
    os.remove(os.path.join(out_dir,'speech_file.xml'))          
    # return number of segments
    return segs.shape[1]
    
# Determines candidate names for given file
def get_context(speech_file,cands_file):
    # pull out filename and folder
    pat = re.compile(r"""
    (?:.*?(?:/|\\))?                 # relative path (optional)
    (?:(?P<state>\w+)_\w+(?:/|\\))?  # folder with format state_name (optional)
    (?P<fn>[^/\\]+)[.]wav$           # file name (required)
    """,re.VERBOSE)
    
    # state (in lowercase 2-letter abbreviation)
    state = pat.match(speech_file).group('state').lower()
    
    # get candidate names
    with open(cands_file,'r') as csv_file:
        reader = csv.DictReader(csv_file)
        cands = [(row['Rep'], row['Dem'], row['Third'])
                        for row in reader if row['State'].lower() == state]

    # error checking
    assert len(cands) == 1,"Incorrect number of entries for state"                    
    # pull out names
    rep,dem,third = cands[0]
    # split names (last names only)
    p = re.compile(r'\W+')
    rep = p.split(rep)[0]
    dem = p.split(dem)[0]
    third = p.split(third)[0]
    # build context
    context = filter(None,[rep,dem,third])
    
    return context

def main(speech_file,cands='',out_dir=''):
    """Transcribe the given audio file.

    Args:
        speech_file: the name of the audio file (in wav format).
        cands: candidate names file (optional)
        out_dir: where the transcripts are stored,  (optional)
    """
    # directory of speech file
    base_dir = os.path.dirname(speech_file)
    # filename
    fname = os.path.basename(speech_file)
    # working directory
    wd = os.getcwdu()
    # get context    
    if cands:
        context = get_context(speech_file,cands)
    else:
        context = []
        
    # diarize
    n = diarization(speech_file)
    
    # output transcript and info file
    if os.path.isabs(out_dir): # absolute path specified
        out_txt = os.path.join(out_dir,fname[:-4]+'.txt')
        info_txt = os.path.join(out_dir,fname[:-4]+'_info.txt')
    else: # relative path (to working directory) specified
        out_txt = os.path.join(wd,out_dir,fname[:-4]+'.txt')
        info_txt = os.path.join(wd,out_dir,fname[:-4]+'_info.txt') 
        
    
    if os.path.isfile(out_txt):
        os.remove(out_txt)
    if os.path.isfile(info_txt):
        os.remove(info_txt)
    
    for i in range(n):
        # upload audio content
        cur_file = speech_file[:-4] + '_' + str(i) + '.raw'
        fn = os.path.basename(cur_file)
        with open(cur_file,'rb') as sf:
            dst_uri = boto.storage_uri(BUCKET + '/' + fn, GOOGLE_STORAGE)
            dst_uri.new_key().set_contents_from_file(sf)
        
        # uri
        uri = boto.storage_uri(BUCKET, GOOGLE_STORAGE)
        gcs_uri = 'gs://' + BUCKET + '/' + fn
        
        # build request
        speech_client = speech.Client()
        audio_sample = speech_client.sample(
            content=None,
            source_uri=gcs_uri,
            encoding='LINEAR16',
            sample_rate_hertz=16000)
        
        # send asynchronous request
        operation = audio_sample.long_running_recognize('en-US',
                                                      speech_contexts = context)
    
        # wait for response
        retry_count = 100
        while retry_count > 0 and not operation.complete:
            retry_count -= 1
            time.sleep(2)
            operation.poll()

        if not operation.complete:
            print('Operation not complete and retry limit reached.')
            return
    
        # delete uploaded data
        for obj in uri.get_bucket():
            obj.delete()
    
        # write confidence scores to output directory
        with open(info_txt,'a') as info:
            for result in operation.results:
                # get transcript chunk length
                resp = result.transcript
                nwords = len(resp.split())
                conf = result.confidence
                # write results to file
                towrite = '{0:3d}, {1:10.8f}\n'.format(nwords,conf)
                info.write(towrite)
    
        # write transcript to file in output directory
        with open(out_txt,'a') as tf:
            txt = ''
            for result in operation.results:
                txt += result.transcript + ' ' 
            tf.write(txt)
    
    # clean up text file
    with open(out_txt,'r+') as fn:
        text = fn.read()
        text = ' '.join(text.split())
        text = wrap(text,width=80)
        fn.seek(0)
        fn.write('\n'.join(text))
        fn.truncate()
            
    # clean up intermediate files
    if base_dir == '':
        base_dir = '.'
    for item in os.listdir(base_dir):
        if item.endswith('.raw'):
            os.remove(os.path.join(base_dir,item))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'speech_file', help='Full path of audio file to be recognized')
    parser.add_argument(
        '-cf','--cands_file', help='Full path of candidate names file for context')
    parser.add_argument(
        '-o','--output', help='Full path of output directory')
    args = parser.parse_args()
    # get arguments
    speech_file = args.speech_file
    
    # check output directory
    if args.output:
        output = args.output
    else:
        output = os.path.dirname(speech_file)
    
    # check candidate file    
    if args.cands_file:
        cands_file = args.cands_file
        main(speech_file,cands_file,output)
    else:
        main(speech_file,out_dir=output)

