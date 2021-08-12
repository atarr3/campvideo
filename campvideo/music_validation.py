import numpy as np
import os
import pandas as pd

from campvideo.audio import Audio
from os.path import basename, exists, join
from sklearn.metrics import confusion_matrix

VID_PATH = 'E:/Users/Alex/Desktop/ITH_Matched'
WMP_PATH = 'E:/Users/Alex/Desktop/wmp_final.csv'
RES_PATH = 'E:/Users/Alex/Desktop/wmp_pred.csv'

# wmp file
wmp = pd.read_csv(WMP_PATH).sort_values('uid')

# list of video files
fpaths = [join(root, fname) for root, _, fnames in os.walk(VID_PATH)
                                for fname in fnames
                                    if fname.lower().endswith(
                                            ('.mp4','.wmv')
                                            )]
fpaths = sorted(fpaths, key=lambda x: basename(x))

# process videos
n = len(fpaths)
m1, m2, m3, m4 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

for i, fpath in enumerate(fpaths):
    # audio class constructor
    aud = Audio(fpath)
    # music mood predictions
    m1[i], m2[i], m3[i] = aud.musicmood()
    m4[i], _ = aud.musicmood(combine_negative=True)

# save results    
if not exists(RES_PATH):
    wmp_pred = pd.DataFrame({'music1' : m1, 'music2' : m2, 
                             'music3' : m3, 'music4' : m4}, index=wmp.uid)
else:
    wmp_pred = pd.read_csv(RES_PATH, index_col=0)
    wmp_pred['music1'], wmp_pred['music2'] = m1, m2
    wmp_pred['music3'], wmp_pred['music4'] = m3, m4
# save
wmp_pred.to_csv('E:/Users/Alex/Desktop/wmp_pred.csv', index_label='uid')