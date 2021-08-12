import cv2
import numpy as np
import os
import pandas as pd
import pickle

from face_recognition import load_image_file, face_distance, face_encodings, face_locations
from os.path import basename, dirname, join
from PIL import Image
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from timeit import default_timer

ID_DIR = r'E:\Users\Alex\Desktop\ITH_Senate_Faces'
FACE_PATH = r'E:\Users\Alex\Desktop\ITH_Senate\face_encodings.pkl'
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_final.csv'
FIG_DIR = r'E:\Users\Alex\OneDrive\Documents\Research\campvideo\paper\figs'

# flag for saving results for cleaning or loading for corrected values
SAVE = True

def build_montages(image_list, image_shape, montage_shape):
    """
    --------------------------------------------------------------------------
    author: Kyle Hounslow
    --------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages

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
wmp = pd.read_csv(WMP_PATH, index_col="uid")

# subset down to us senate race entries
wmp_sen = wmp[wmp.race == "US SENATE"][["vid","f_picture","o_picture"]]

# remove videos missing labels for both vid and f_picture or o_picture
wmp_sen = wmp_sen.loc[(~wmp_sen.vid.isna() | ~wmp_sen.f_picture.isna()) &
                      ~wmp_sen.o_picture.isna()]

# process data
n = len(face_encs)
cmin = []
omin = []
y_cand = []
y_opp = []
face_paths = set()
for i, ((metadata, uid), encs) in enumerate(face_encs):
    print("Processing video %d of %d... " % (i+1,n), end='', flush=True)
    s = default_timer()
    
    if not wmp_sen.uid.str.contains(uid).any(): 
        print('\n')
        continue

    # get fpaths for candidate and opponent(s)
    cand_im_fpath = next(filter(lambda x: metadata.lower() in x.lower(),
                                fpaths_id))
    opp_im_fpaths = list(filter(lambda x: dirname(metadata).lower() in x.lower()
                                and basename(metadata).lower() not in x.lower(),
                                fpaths_id))
    # update face_paths object
    face_paths.add(cand_im_fpath)
    [face_paths.add(opp_im_fpath) for opp_im_fpath in opp_im_fpaths]
    
    # load and resize
    cand_im = resize_im(load_image_file(cand_im_fpath))
    opp_ims = [resize_im(load_image_file(fpath)) for fpath in opp_im_fpaths]
    
    # take first face detected
    cand_enc = face_encodings(cand_im,
                              face_locations(cand_im, model='cnn'),
                              model="large",
                              num_jitters=100)[0]
    opp_encs = [face_encodings(opp_im,
                               face_locations(opp_im, model='cnn'),
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
    sub = wmp_sen.loc[uid]
    y_cand.append(np.nanmax((sub.vid, sub.f_picture)))
    y_opp.append(sub.o_picture)
    
    # report time taken
    print("Done in %.1fs" % (default_timer()-s))
    
# create montage
face_paths = sorted(list(face_paths), key=lambda x: basename(x))
# remove duplicate entries (ran in two different races)
face_paths.remove(join(ID_DIR, r'sen2014\NH\brown_R\brown0.jpg'))
face_paths.remove(join(ID_DIR, r'sen2012\DE\wade_R\wade0.jpg'))
ims = []
im_size = (80,80)
for i, fpath in enumerate(face_paths):
    # resize image
    im = resize_im(load_image_file(fpath))
    # detect face
    bb = face_locations(im, model='cnn')[0]
    face = cv2.resize(im[bb[0]:bb[2], bb[3]:bb[1], :], im_size)
    ims.append(face)

montage = build_montages(ims, im_size, (19, 6))[0]
im = Image.fromarray(montage[:,:])
im.save(join(FIG_DIR, 'montage.pdf'))

# convert to numpy arrays
y_cand = np.array(y_cand)
y_opp = np.array(y_opp)
cmin = np.array(cmin)
omin = np.array(omin) 
    
# train/test split
# xc_train,xc_test,yc_train,yc_test = train_test_split(cmin.reshape(-1,1),
#                                                      y_cand,
#                                                      test_size=0.2,
#                                                      random_state=2002)
xo_train,xo_test,yo_train,yo_test = train_test_split(omin.reshape(-1,1),
                                                     y_opp,
                                                     test_size=0.2,
                                                     random_state=2002)

# svm (opponent)
params = [{'C': np.geomspace(0.01, 10, 25)}]
grid_o = GridSearchCV(svm.LinearSVC(dual=False),
                      params, scoring='accuracy', cv=5)
grid_o.fit(xo_train, yo_train)
svm_o = grid_o.best_estimator_

# # svm (candidate)
# params = [{'C': np.geomspace(0.01, 10, 25)}]
# grid_c = GridSearchCV(svm.LinearSVC(dual=False),
#                       params, scoring='accuracy', cv=5)
# grid_c.fit(xc_train, yc_train)
# svm_c = grid_c.best_estimator_

# distance threshold
thr_o = -svm_o.intercept_[0] / svm_o.coef_[0][0]
# thr_c = -svm_c.intercept_[0] / svm_c.coef_[0][0]

# predicted values
o_pred = (omin < thr_o).astype(int)
c_pred = (cmin < thr_o).astype(int)

# metrics
o_p = metrics.precision_score(y_opp, o_pred)
o_r = metrics.recall_score(y_opp, o_pred)
o_f1 = metrics.f1_score(y_opp, o_pred)
o_a = metrics.accuracy_score(y_opp, o_pred)

print(80 * '-')
print("Opponent Mention Results")
print()
print("Distance Threshold = %.4f" % thr_o)
print()
print("Precision: %.4f" % o_p)
print("   Recall: %.4f" % o_r)
print(" F1 Score: %.4f" % o_f1)
print(" Accuracy: %.4f" % o_a)

c_p = metrics.precision_score(y_cand, c_pred)
c_r = metrics.recall_score(y_cand, c_pred)
c_f1 = metrics.f1_score(y_cand, c_pred)
c_a = metrics.accuracy_score(y_cand, c_pred)

print(80 * '-')
print("Candidate Mention Results")
print()
print("Distance Threshold = %.4f" % thr_o)
print()
print("Precision: %.4f" % c_p)
print("   Recall: %.4f" % c_r)
print(" F1 Score: %.4f" % c_f1)
print(" Accuracy: %.4f" % c_a)

if SAVE:
    # save encodings
    with open(join(dirname(FACE_PATH),'cand.pkl'),'wb') as fh:
        pickle.dump(zip(cmin,y_cand), fh)
        
    with open(join(dirname(FACE_PATH),'opp.pkl'),'wb') as fh:
        pickle.dump(zip(omin,y_opp), fh)
        
    # opponent disagreement analysis
    wmp_opp = wmp.loc[wmp_sen.index][["creative","year","o_picture"]]
    wmp_opp = wmp_opp.iloc[wmp_opp.index.str.lower().argsort()]
    wmp_opp["o_picture_pred"] = o_pred
    wmp_opp["omin"] = np.around(omin, 5)
    wmp_opp["dist"] = np.around(np.abs(omin-thr_o), 5)
    
    opp_dis = wmp_opp.iloc[np.where(o_pred != y_opp)[0]]

    # save
    opp_dis.to_csv(join(dirname(WMP_PATH),'opp_dis.csv'), index_label='uid')
    
    # candidate disagreement analysis
    wmp_can = wmp.loc[wmp_sen.index][["creative","year","vid", "f_picture"]]
    wmp_can = wmp_can.iloc[wmp_can.index.str.lower().argsort()]
    wmp_can["f_picture_OR"] = np.nanmax((wmp_can.vid, wmp_can.f_picture), axis=0)
    wmp_can["f_picture_pred"] = c_pred
    wmp_can["cmin"] = np.around(cmin, 5)
    wmp_can["dist"] = np.around(np.abs(cmin-thr_o), 5)
    
    cand_dis = wmp_can.iloc[np.where(c_pred != y_cand)[0]]

    # save
    cand_dis.to_csv(join(dirname(WMP_PATH),'cand_dis.csv'), index_label='uid')

else:
    # load in data
    with open(join(dirname(FACE_PATH),'cand.pkl'),'rb') as fh:
        cmin, y_cand = [np.array(item) for item in zip(*pickle.load(fh))]
        
    with open(join(dirname(FACE_PATH),'opp.pkl'),'rb') as fh:
        omin, y_opp = [np.array(item) for item in zip(*pickle.load(fh))]
    
    # opponent corrections
        
    opp_dis = pd.read_csv(join(dirname(WMP_PATH),'opp_dis_correct.csv'), 
                          index_col="uid")
    fix = opp_dis.loc[opp_dis.note == 'wmp wrong'].index
    print("WMP Error Rate for o_picture: %.2f" % (len(fix) / len(opp_dis)))
    
    # load in wmp file and subset to senate data
    wmp = pd.read_csv(WMP_PATH, index_col="uid")
    wmp_sen = wmp[wmp.race == "US SENATE"][["vid","f_picture","o_picture"]] 
    # remove videos missing labels for both vid and f_picture or o_picture
    wmp_sen = wmp_sen.loc[(~wmp_sen.vid.isna() | ~wmp_sen.f_picture.isna()) &
                          ~wmp_sen.o_picture.isna()]

    # corrected labels variable
    wmp_sen['o_picture_corr'] = wmp_sen.o_picture
    # invert labels
    replace = (wmp_sen.loc[fix].o_picture == 0).astype(float)
    wmp_sen.loc[fix, "o_picture_corr"] = replace
    y_oppc = wmp_sen.iloc[wmp_sen.index.str.lower().argsort()].o_picture_corr
    
    # candidate corrections
    
    cand_dis = pd.read_csv(join(dirname(WMP_PATH),'cand_dis_correct.csv'), 
                           index_col="uid")
    fix = cand_dis.loc[cand_dis.note == 'wmp wrong'].index
    defi = cand_dis.loc[cand_dis.note == 'definition'].index
    print("WMP Error Rate for f_picture: %.2f" % (len(fix) / len(cand_dis)))
    print("WMP Definition Rate for f_picture: %.2f" % (len(defi) / len(cand_dis)))
    
    wmp_sen['f_picture_corr'] = np.nanmax((wmp_sen.vid, wmp_sen.f_picture), axis=0)
    # invert labels
    replace = (wmp_sen.loc[fix.append(defi)].f_picture_corr == 0).astype(float)
    wmp_sen.loc[fix.append(defi), "f_picture_corr"] = replace
    y_candc = wmp_sen.iloc[wmp_sen.index.str.lower().argsort()].f_picture_corr

    # fit SVM to corrected data
    xo_train,xo_test,yo_train,yo_test = train_test_split(omin.reshape(-1,1),
                                                         y_oppc,
                                                         test_size=0.2,
                                                         random_state=2002)
    
    params = [{'C': np.geomspace(0.01, 10, 25)}]
    grid_o = GridSearchCV(svm.LinearSVC(dual=False),
                          params, scoring='accuracy', cv=5)
    grid_o.fit(xo_train, yo_train)
    svm_o = grid_o.best_estimator_
    
    # distance threshold
    thr_o = -svm_o.intercept_[0] / svm_o.coef_[0][0]
    # predicted values
    o_pred = (omin < thr_o).astype(int)
    c_pred = (cmin < thr_o).astype(int)

    # metrics
    o_p = metrics.precision_score(y_oppc, o_pred)
    o_r = metrics.recall_score(y_oppc, o_pred)
    o_f1 = metrics.f1_score(y_oppc, o_pred)
    o_a = metrics.accuracy_score(y_oppc, o_pred)
    
    print(80 * '-')
    print("Opponent Mention Results (Corrected)")
    print()
    print("Distance Threshold = %.4f" % thr_o)
    print()
    print("Precision: %.4f" % o_p)
    print("   Recall: %.4f" % o_r)
    print(" F1 Score: %.4f" % o_f1)
    print(" Accuracy: %.4f" % o_a)
    
    c_p = metrics.precision_score(y_candc, c_pred)
    c_r = metrics.recall_score(y_candc, c_pred)
    c_f1 = metrics.f1_score(y_candc, c_pred)
    c_a = metrics.accuracy_score(y_candc, c_pred)
    
    print(80 * '-')
    print("Candidate Mention Results (Corrected)")
    print()
    print("Distance Threshold = %.4f" % thr_o)
    print()
    print("Precision: %.4f" % c_p)
    print("   Recall: %.4f" % c_r)
    print(" F1 Score: %.4f" % c_f1)
    print(" Accuracy: %.4f" % c_a)
    