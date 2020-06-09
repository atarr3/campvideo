import numpy as np
import os
import pandas as pd
import pickle

from cv2 import resize
from face_recognition import load_image_file,face_distance,face_encodings,face_locations
from os.path import basename,dirname,join
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from timeit import default_timer

ID_DIR = r'E:\Users\Alex\Desktop\ITH_Senate_Faces'
FACE_PATH = r'E:\Users\Alex\Desktop\ITH_Senate\face_encodings.pkl'
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_english_nodup.csv'

# function for resizing image while preserving aspect-ratio
def resize_im(im,max_dim=1280):
    h,w,_ = im.shape
    # dont resize if image is below the maximum dimension
    if max(h,w) <= max_dim: 
        return im
    
    # compute aspect-ratio
    ar = w / h
    # scale to max dim
    new_w = max_dim if w >= h else int(max_dim * ar)
    new_h = max_dim if w < h else int(max_dim / ar)
    
    # return resized image
    return resize(im,(new_w,new_h))
        
# get list of ids
fpaths_id = [join(root,fname) for root,_,fnames in os.walk(ID_DIR)
                                   for fname in fnames
                                       if fname.lower().endswith(
                                               ('.jpg','.jpeg','.png')
                                               )]

# open face encodings and metadata
with open(FACE_PATH,'rb') as fh:
    face_encs = pickle.load(fh)

# open wmp file
wmp = pd.read_csv(WMP_PATH)

# subset down to us senate race entries
wmp_sen = wmp[wmp.race == "US SENATE"][["uid","vid","f_picture","o_picture"]]

# process data
n = len(face_encs)
cmin = []
omin = []
y_cand = []
y_opp = []
for i, ((metadata, uid), encs) in enumerate(face_encs):
    print("Processing video %3d of %d.... " % (i+1,n),end='',flush=True)
    s = default_timer()

    # get fpaths for candidate and opponent(s)
    cand_im_fpath = next(filter(lambda x: metadata.lower() in x.lower(),
                                fpaths_id))
    opp_im_fpaths = list(filter(lambda x: dirname(metadata).lower() in x.lower()
                                and basename(metadata).lower() not in x.lower(),
                                fpaths_id))
    # load and resize
    cand_im = resize_im(load_image_file(cand_im_fpath))
    opp_ims = [resize_im(load_image_file(fpath)) for fpath in opp_im_fpaths]
    
    # take first face detected
    cand_enc = face_encodings(cand_im,
                              face_locations(cand_im,model='cnn'),
                              model="large",
                              num_jitters=100)[0]
    opp_encs = [face_encodings(opp_im,
                               face_locations(opp_im,model='cnn'),
                               model="large",
                               num_jitters=100)[0] for opp_im in opp_ims]

    # compute minimum distances
    cdist = face_distance(encs, cand_enc)
    odists = [face_distance(encs, opp_enc) for opp_enc in opp_encs]

    if len(encs) > 0:
        cmin.append(min(cdist))
        omin.append(min([min(odist) for odist in odists]))
    else:
        # no faces detected, append maximum distance
        cmin.append(1)
        omin.append(1)
        
    # get wmp appearance labels
    sub = wmp_sen.loc[wmp.uid == uid]
    y_cand.append(int(sub.vid) or int(sub.f_picture))
    y_opp.append(int(sub.o_picture))
    
    # report time taken
    print("Done in %.1fs" % (default_timer()-s))

# convert to numpy arrays
y_cand = np.array(y_cand)
y_opp = np.array(y_opp)
cmin = np.array(cmin)
omin = np.array(omin)

# save
with open(join(dirname(FACE_PATH),'cand.pkl'),'wb') as fh:
    pickle.dump(zip(cmin,y_cand), fh)
    
with open(join(dirname(FACE_PATH),'opp.pkl'),'wb') as fh:
    pickle.dump(zip(omin,y_opp), fh)
    
# load
with open(join(dirname(FACE_PATH),'cand.pkl'),'rb') as fh:
    cmin, y_cand = [np.array(item) for item in zip(*pickle.load(fh))]
    
with open(join(dirname(FACE_PATH),'opp.pkl'),'rb') as fh:
    omin, y_opp = [np.array(item) for item in zip(*pickle.load(fh))]
    
    
# train/test split
xc_train,xc_test,yc_train,yc_test = train_test_split(cmin.reshape(-1,1),
                                                     y_cand,
                                                     test_size=0.2,
                                                     random_state=2002)
xo_train,xo_test,yo_train,yo_test = train_test_split(omin.reshape(-1,1),
                                                     y_opp,
                                                     test_size=0.2,
                                                     random_state=2002)

# svm (opponent)
svc_o = svm.LinearSVC(C=1,dual=False)
svc_o.fit(xo_train,yo_train)
svc_o.score(xo_test,yo_test)

# distance threshold
thr = -svc_o.intercept_[0] / svc_o.coef_[0][0]

# predicted values
o_pred = (omin < thr).astype(int)

# metrics
o_p = metrics.precision_score(y_opp,o_pred)
o_r = metrics.recall_score(y_opp,o_pred)
o_f1 = metrics.f1_score(y_opp,o_pred)
o_a = metrics.accuracy_score(y_opp,o_pred)

print(80 * '-')
print("Opponent Mention Results")
print()
print("Distance Threshold = %.4f" % thr)
print()
print("Precision: %.4f" % o_p)
print("   Recall: %.4f" % o_r)
print(" F1 Score: %.4f" % o_f1)
print(" Accuracy: %.4f" % o_a)

# disagreement analysis (save)
# wmp_opp = wmp[wmp.race == "US SENATE"][["uid","creative","year","o_picture"]]
# wmp_opp = wmp_opp.iloc[wmp_opp.uid.str.lower().argsort()].astype({'o_picture' : int})
# wmp_opp.index = range(len(wmp_opp))
# wmp_opp["o_picture_pred"] = o_pred
# wmp_opp["omin"] = np.around(omin,5)
# wmp_opp["dist"] = np.around(np.abs(omin-thr),5)

# dis = wmp_opp.iloc[np.where(o_pred != y_opp)[0]]

# # save
# dis.to_csv(join(dirname(WMP_PATH),'opp_dis.csv'),index=False)

# disagreement analysis (load)
dis = pd.read_csv(join(dirname(WMP_PATH),'opp_dis.csv'))
names = dis[dis.note == 'wmp wrong'].uid
print("WMP Error Rate: %.2f" % (len(names)/len(dis)))
# corrected
# open wmp file
wmp = pd.read_csv(WMP_PATH)
# subset down to us senate race entries
wmp_sen = wmp[wmp.race == "US SENATE"][["uid","vid","f_picture","o_picture"]]
# corrected labels
wmp_sen['o_picture_corr'] = wmp_sen.o_picture
wmp_sen.loc[wmp_sen.uid.isin(names),'o_picture_corr'] = np.logical_xor(True,
                                wmp_sen.loc[wmp_sen.uid.isin(names)].o_picture_corr
                                ).astype(float)
y_oppc = wmp_sen.iloc[wmp_sen.uid.str.lower().argsort()].o_picture_corr.to_numpy()

# metrics
o_p = metrics.precision_score(y_oppc,o_pred)
o_r = metrics.recall_score(y_oppc,o_pred)
o_f1 = metrics.f1_score(y_oppc,o_pred)
o_a = metrics.accuracy_score(y_oppc,o_pred)

print(80 * '-')
print("Opponent Mention Results (Corrected)")
print()
print("Distance Threshold = %.4f" % thr)
print()
print("Precision: %.4f" % o_p)
print("   Recall: %.4f" % o_r)
print(" F1 Score: %.4f" % o_f1)
print(" Accuracy: %.4f" % o_a)
    