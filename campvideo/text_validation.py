import cv2
import numpy as np
import os
import pandas as pd

from campvideo import Keyframes,Text
from face_recognition import face_encodings,face_locations,load_image_file
from os.path import basename,dirname,join
from pkg_resources import resource_filename
from sklearn.metrics import confusion_matrix

# function for resizing image while preserving aspect-ratio
def resize_im(im,max_dim=1280):
    h, w,_ = im.shape
    # dont resize if image is below the maximum dimension
    if max(h, w) <= max_dim: 
        return im
    
    # compute aspect-ratio
    ar = w / h
    # scale to max dim
    new_w = max_dim if w >= h else int(max_dim * ar)
    new_h = max_dim if w < h else int(max_dim / ar)
    
    # return resized image
    return cv2.resize(im, (new_w,new_h))

# function for creating list of names to check in name mentions
def namegen(name, return_plurals=True):
    # name is a single string for the full name, no punctuation removed
    
    # return None if nan passed
    if type(name) is not str: return [None]
    
    # split names, ignore first name
    names = name.split(' ')[1:]
    
    # add hyphen to double last name
    name_hy = '-'.join(names) if len(names) == 2 else None
    
    # remove hyphen from names
    names_nohy = [name.split('-') if '-' in name else None for name in names]
    names_nohy = [' '.join(item) for item in names_nohy if item is not None]
    
    # final part of name for double last name
    if len(names) == 2:
        final = names[-1]
    elif '-' in names[0]:
        final = names[0].split('-')[-1]
    else:
        final = None
        
    # preliminary list
    gen_names = list(filter(None, [' '.join(names)] + [name_hy] + 
                                  names_nohy + [final]))
        
    # add pluralization (this seems like overkill on June's part)
    plurals = [item + 's' for item in gen_names if not item.endswith('s')]
    
    # NOTE: no need to add possessive, spacy tokenizes 's separately
    
    if return_plurals:
        return gen_names + plurals
    else:
        return gen_names
    
# function for cleaning wmp data
def clean_wmp(data):
    # merge issue30 (abortion) and issue58 (women's health)
    data['issue30'] = np.logical_or(data.issue30, data.issue58).astype(int)
    # merge issue53 (healthcare) and issue59 (obamacare)
    # NOTE: issue59 only labeled in 2014 data
    data['issue53'] = np.logical_or(data.issue53, data.issue59, 
                                    out=data.issue53.to_numpy(),
                                    where=~data.issue59.isna())
    
    # subset to correct columns
    issues = data.loc[:, data.columns.isin(VOCAB.wmp)]
    
    # convert non-binary variables to binary
    for issue, column  in issues.iteritems():
        if 'mention' in issue or 'issue' in issue: 
            continue
        issues[issue] = (~np.logical_or(column == '0', column == 'No')).astype(int)
    
    # return
    return issues

# function for creating dataframes from issue vectors
def make_frames(issue_trans, issue_im, uid):
    # make dataframes
    data_trans = pd.concat(issue_trans, axis=1).T
    data_im = pd.concat(issue_im, axis=1).T
    
    # rename columns to WMP labels
    data_trans.columns = VOCAB.wmp
    data_im.columns = VOCAB.wmp
    
    # set index to uid
    data_trans.rename_axis("", axis=1, inplace=True)
    data_trans.index = uid
    data_im.rename_axis("", axis=1, inplace=True)
    data_im.index = uid
    
    return data_trans, data_im

# issue vocabulary list
VOCAB_PATH = resource_filename('campvideo','data/issuenames.csv')
VOCAB = pd.read_csv(VOCAB_PATH)

# picture of obama
OBAMA_PATH = resource_filename('campvideo','data/obama0.jpg')
SAVE = False # flag for saving face recognition results or loading in

# face data
FACE_PATH = r'E:\Users\Alex\Desktop\ITH_Senate\face_encodings.pkl'

# image text and keyframes
VDATA_DIR = r'E:\Users\Alex\Desktop\ITH_Matched_summaries'

# video directory
VID_DIR = r'E:\Users\Alex\Desktop\ITH_Matched'

# transcripts
TRAN_DIR = r'E:\Users\Alex\Desktop\ITH_Matched\transcripts'

# wmp data
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_final.csv'
wmp = pd.read_csv(WMP_PATH, index_col='uid')
wmp = wmp.iloc[wmp.index.str.lower().argsort()]

# June's files
JUNE_DIR = r'E:\Users\Alex\OneDrive\Documents\Research\campvideo\U2D Documents\Source'

# opponent names
META_PATH = r'E:\Users\Alex\Desktop\metadata.csv'
meta = pd.read_csv(META_PATH, index_col='uid')
meta = meta.iloc[meta.index.str.lower().argsort()]

# sorted transcript paths
tpaths = [join(root,fname) for root,_,fnames in os.walk(TRAN_DIR)
                               for fname in fnames
                                   if fname.lower().endswith('.txt')]
tpaths = sorted(tpaths, key=lambda x: basename(x).lower())

# sorted image text paths
ipaths = [join(root,fname) for root,_,fnames in os.walk(VDATA_DIR)
                               for fname in fnames
                                   if fname.lower() == 'image_text.txt']
ipaths = sorted(ipaths, key=lambda x: basename(dirname(x)).lower())

# sorted summary paths
spaths = [join(root,fname) for root,_,fnames in os.walk(VDATA_DIR)
                               for fname in fnames
                                   if fname.lower() == 'keyframes.txt']
spaths = sorted(spaths, key=lambda x: basename(dirname(x)).lower())

# sorted video paths
vpaths = [join(root,fname) for root,_,fnames in os.walk(VID_DIR)
                               for fname in fnames
                                   if fname.endswith('.mp4')]
vpaths = sorted(vpaths, key=lambda x: basename(x).lower())

# number of samples
n = len(tpaths)
  
# get obama face encoding
obama_im = resize_im(load_image_file(OBAMA_PATH))
obama_enc = face_encodings(obama_im,
                           face_locations(obama_im, model='cnn'),
                           model="large",
                           num_jitters=100)[0]

# prediction vectors
obama_im = []
issue_trans = []
issue_im = []
opp_trans = []
opp_im = []

# predict labels
for i, (vpath, spath, tpath, ipath) in enumerate(zip(vpaths, spaths, tpaths, ipaths)):
    print('Processing video %d of %d...' % (i+1, n))
    # get opponent last name(s)
    opp_ma = meta.iloc[i].opp_major
    opp_th = meta.iloc[i].opp_third
    
    # preprocess because text is horrible
    names_ma = namegen(opp_ma)
    names_th = namegen(opp_th)

    # single list of opponents
    opps = list(filter(None, names_ma + names_th))
    
    # read in transcript, consruct Text object
    with open(tpath, 'r', encoding='utf8') as fh:
        transcript = fh.read().lower()
        
    trans_text = Text(transcript)
    
    # read in image text, construct Text object
    with open(ipath, 'r', encoding='utf8') as fh:
        # formatting in these files is weird
        image_text = fh.read().lower().strip('\n').replace('\n', ' '
                      ).strip('\t').replace('\t', ' '
                      ).replace('.', ' ') # deals with .com
    
    im_text = Text(image_text)
    
    # read in summary and construct Keyframes object
    if SAVE:
        with open(spath, 'r') as fh:
            kf_ind = fh.read().split(',')
            kf_ind = [int(item) for item in kf_ind]
            
        kf = Keyframes.fromvid(vpath, kf_ind)
        
        # face rec
        obama_im.append(kf.facerec(obama_enc, dist_thr=0.5156))
    
    # issue mention (transcript)
    issue_trans.append(
        trans_text.issue_mention(include_names=True, include_phrases=True)
        )
    # issue mention (image text)
    issue_im.append(
        im_text.issue_mention(include_names=True, include_phrases=True)
        )
    
    # opponent mention
    opp_trans.append(any(trans_text.opp_mention(name) for name in opps)) 
    opp_im.append(any(im_text.opp_mention(name) for name in opps))

# save/load face recognition results for obama
if SAVE:
    data_o = pd.DataFrame(data={'obama_im' : obama_im}, dtype=int, 
                          index=wmp.index)
    data_o.to_csv(join(dirname(WMP_PATH), 'obama_im.csv'), index_label='uid')
else:
    data_o = pd.read_csv(join(dirname(WMP_PATH), 'obama_im.csv'), 
                         index_col='uid')
    
# compare against wmp
issues_wmp = clean_wmp(wmp)
issues_trans, issues_im = make_frames(issue_trans, issue_im, wmp.index)
# face recognition for obama variable (`prsment`)
issues_im['prsment'] = issues_im.prsment | data_o.obama_im
# ignore visual data for congress (`congmt`)  and wall street (`mention16`)
issues_im['congmt'] = 0
issues_im['mention16'] = 0

opp_wmp = wmp.loc[~wmp.o_mention.isna()].o_mention
opp_trans = pd.Series(opp_trans, name='o_mention', dtype=int, index=wmp.index)
opp_im = pd.Series(opp_im, name='o_mention', dtype=int, index=wmp.index)

# confusion matrices
npairs = issues_wmp.size
nopp = len(opp_wmp)
# transcript only
c_issues_t = confusion_matrix(issues_wmp.values.ravel(), 
                              issues_trans.values.ravel())
c_opp_t = confusion_matrix(opp_wmp, opp_trans.loc[~wmp.o_mention.isna()])
# transcript + visual data
c_issues_tv = confusion_matrix(issues_wmp.values.ravel(), 
                               (issues_trans | issues_im).values.ravel())
c_opp_tv = confusion_matrix(opp_wmp, 
                            (opp_trans | opp_im).loc[~wmp.o_mention.isna()])

# print results
print('Issue Mention (Audio Only):')
print()
print(c_issues_t)
print()
print(c_issues_t / npairs)
print()

print('Issue Mention (Audio + Visual):')
print()
print(c_issues_tv)
print()
print(c_issues_tv / npairs)
print()

print('Opponent Mention (Audio Only):')
print()
print(c_opp_t)
print()
print(c_opp_t / nopp)
print()

print('Opponent Mention (Audio + Visual):')
print()
print(c_opp_tv)
print()
print(c_opp_tv / nopp)
print()

# opponent mention validation
opp_pred = (opp_trans | opp_im).loc[~wmp.o_mention.isna()]
ind = (opp_pred != opp_wmp)
opp_val = pd.DataFrame({'wmp' : opp_wmp.loc[ind], 'pred' : opp_pred.loc[ind]}
                      ).sort_index(level='uid', key=lambda x: x.str.lower()
                      ).reindex(columns=['wmp', 'pred', 'correct', 'note']
                      ).to_csv(join(dirname(WMP_PATH), 'opp_val_sample.csv'), 
                               index_label='uid')

# disagreement samples
issues_pred = (issues_trans | issues_im)
vid, issue = np.where(issues_wmp != issues_pred)

vi_pairs = pd.DataFrame({'uid' : issues_wmp.index[vid], 
                         'issue' : VOCAB.wmp[issue], 
                         'desc' : VOCAB.desc[issue],
                         'wmp' : issues_wmp.values[vid, issue],
                         'pred' : issues_pred.values[vid, issue]}
                        ).set_index(['uid', 'issue'])

# read in June's weird files and construct single data frame
suff = ('15_f.csv', '15_tn.csv', '15_tp.csv', '30_f.csv', '30_tn.csv',
        '30_tp.csv', '60_f.csv', '60_tn.csv', '60_tp.csv')
data = pd.DataFrame()
for fname in os.listdir(JUNE_DIR):
    if fname.endswith(suff):    
        data = pd.concat([data,
                          pd.read_csv(join(JUNE_DIR, fname), index_col='uid',
                          usecols=['uid', 'desc', 'wmp', 'wmpcode', 'ithcode']
                          )])
# dataframe cleanup
data = data.rename(columns={'wmp' : 'issue', 'wmpcode' : 'wmp', 
                            'ithcode' : 'pred'}              # rename columns
                  ).set_index('issue', append=True           # add issue index
                  ).sort_index(level=['uid', 'issue'],       # sort index
                               key=lambda x: x.str.lower()
                  )
# subset down to videos in our sample (why are there primary videos????)
sub = data.loc[data.index.get_level_values('uid').isin(issues_wmp.index)]                            

# correct incorrect wmp data (why is this wrong June?????)
sub['wmp'] = np.diag(issues_wmp.loc[sub.index.get_level_values('uid'), 
                                    sub.index.get_level_values('issue')
                                   ].values)
# update predictions
sub['pred'] = np.diag(issues_pred.loc[sub.index.get_level_values('uid'), 
                                      sub.index.get_level_values('issue')
                                     ].values)

# remove duplicates (why are there duplicates?????????????)
sub = sub.loc[~sub.index.duplicated()]
        
# subset down to 200 'correct' and 300 'incorrect' samples (new MTurk sample)
incorrect = sub.loc[sub.wmp != sub.pred]
correct = sub.loc[sub.wmp == sub.pred]

mt_inc = incorrect.sample(n=300, replace=False, random_state=2002)
mt_cor = correct.sample(n=200, replace=False, random_state=2002)
mt = pd.concat([mt_inc, mt_cor]
              ).sort_index(level=['uid', 'issue'], key=lambda x: x.str.lower()
              )

# MTurk data
mturk = pd.read_csv(join(JUNE_DIR, 'm2_compiled.csv'), 
                    usecols=['Input.uid', 'Input.desc', 'Input.wmp', 'Answer.Q1']
                ).rename(columns={'Input.desc' : 'desc', 'Input.uid' : 'uid', 
                                  'Input.wmp' : 'issue', 'Answer.Q1' : 'mt_pred'}
                ).set_index(['uid', 'issue']
                ).sort_index(level=['uid', 'issue'], key=lambda x: x.str.lower()
                )

# cast Y/N to 1/0
mturk['mt_pred'] = (mturk.mt_pred == 'Y').astype(int)

# remove duplicates subset to samples in modified MTurk sample
mturk = mturk.iloc[np.repeat(~mturk.iloc[::5].index.duplicated(), 5)]
mturk = mturk.loc[mturk.index.isin(mt.index)]

# add columns for wmp and our predictions
mturk['wmp'] = np.repeat(mt.wmp, 5)
mturk['pred'] = np.repeat(mt.pred, 5)

# save
mturk.to_csv(join(dirname(WMP_PATH), 'issues_mturk.csv'), 
             index_label=['uid', 'issue'])

# validation sample
val_fp = mt_inc.loc[mt_inc.pred == 1].sample(70, random_state=2002)
val_fn = mt_inc.loc[mt_inc.pred == 0].sample(50, random_state=2002)
val = pd.concat([val_fn, val_fp]
               ).sort_index(level=['uid', 'issue'], key=lambda x: x.str.lower()
               ).reindex(columns=['desc', 'wmp', 'pred', 'correct', 'note']
               ).to_csv(join(dirname(WMP_PATH), 'issue_val_sample.csv'), 
                        index_label=['uid', 'issue'])
