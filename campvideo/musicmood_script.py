import numpy as np
import pandas as pd
import warnings

from collections import Counter
from numpy.random import choice
from os.path import join
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

FEAT_PATH = r'E:\Users\Alex\Desktop\ITH_Matched'
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_final.csv'
MT_DIR = r'E:\Users\Alex\OneDrive\Documents\Research\campvideo\U2D Documents\Source'
SEED = 2002

# annoying sklearn warning suppression
warnings.filterwarnings('ignore') 

# read in wmp data
wmp = pd.read_csv(WMP_PATH,index_col='uid',
                  usecols=['uid','music1','music2','music3']
                  ).sort_index(
                  ).astype(int)

# find entries which have music (at least one nonzero)
music_ind = (wmp.music1 != 0) | (wmp.music2 != 0) | (wmp.music3 != 0)
music = wmp[music_ind]

# read in features and subset
# data_all = pd.read_csv(join(FEAT_PATH,'mfeats_all.csv'),index_col=0)[music_ind]
data_best = pd.read_csv(join(FEAT_PATH,'mfeats_best.csv'),index_col=0)[music_ind]

# classification pipelines
pipe1 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',SVC(kernel='rbf',class_weight='balanced'))
                  ])
pipe2 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',SVC(kernel='rbf',class_weight='balanced'))
                  ])
pipe3 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',SVC(kernel='rbf',class_weight='balanced'))
                  ])
pipe4 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',SVC(kernel='rbf',class_weight='balanced'))
                  ])

# grid search for tuning parameters, optimizing accuracy or balanced accuracy
params = [{'clf__gamma': np.logspace(-5, -2, 16),
           'clf__C': np.geomspace(0.1, 10, 16)}]

#####################################
## performance using best features ##
#####################################

# # train/test splits
# x1_train,x1_test,y1_train,y1_test = train_test_split(data_best.values,
#                                                      music.music1,
#                                                      test_size=0.2,
#                                                      random_state=SEED)
# x2_train,x2_test,y2_train,y2_test = train_test_split(data_best.values,
#                                                      music.music2,
#                                                      test_size=0.2,
#                                                      random_state=SEED)
# x3_train,x3_test,y3_train,y3_test = train_test_split(data_best.values,
#                                                      music.music3,
#                                                      test_size=0.2,
#                                                      random_state=SEED)
# # combining music1 with music3
# x4_train,x4_test,y4_train,y4_test = train_test_split(data_best.values,
#                                                      np.maximum(music.music1,
#                                                                 music.music3),
#                                                      test_size=0.2,
#                                                      random_state=SEED)

# train/test splits, based of MTurk validation study videos
np.random.seed(SEED)
t1 = pd.read_csv(join(MT_DIR, 'music_result_15.csv'))
t2 = pd.read_csv(join(MT_DIR, 'music_result_30.csv'))
t3 = pd.read_csv(join(MT_DIR, 'music_result_60.csv'))

# filter t2 down to videos which had 5 MTurk responses
counts = Counter(t2["Input.uid"])
keep_id = [key for key, val in zip(counts.keys(), counts.values()) if val == 5]
t2 = t2.loc[t2["Input.uid"].isin(keep_id)]

# get test video ids
t1_test = choice(music.loc[music.index.isin(t1["Input.uid"])].index, 6, 
                 replace=False)
t2_test = choice(music.loc[music.index.isin(t2["Input.uid"])].index, 432,
                 replace=False)
t3_test = choice(music.loc[music.index.isin(t3["Input.uid"])].index, 12,
                 replace=False)
test_ids = np.sort(np.concatenate((t1_test, t2_test, t3_test)))

# get training data
x1_train = data_best.loc[~data_best.index.isin(test_ids)]
x2_train = x3_train = x4_train = x1_train

music_tr = music.loc[~music.index.isin(test_ids)]
y1_train, y2_train, y3_train = music_tr.music1, music_tr.music2, music_tr.music3
y4_train = np.maximum(music_tr.music1, music_tr.music3)

# get test data
x1_test = data_best.loc[data_best.index.isin(test_ids)]
x2_test = x3_test = x4_test = x1_test

music_te = music.loc[music.index.isin(test_ids)]
y1_test, y2_test, y3_test = music_te.music1, music_te.music2, music_te.music3
y4_test = np.maximum(music_te.music1, music_te.music3)

# fit classifiers
clf_a1 = GridSearchCV(pipe1,params,scoring='accuracy',cv=5,verbose=2)
clf_a2 = GridSearchCV(pipe2,params,scoring='accuracy',cv=5,verbose=2)
clf_ba3 = GridSearchCV(pipe3,params,scoring='balanced_accuracy',cv=5,verbose=2)
clf_a4 = GridSearchCV(pipe4,params,scoring='accuracy',cv=5,verbose=2)

clf_a1.fit(x1_train,y1_train)
clf_a2.fit(x2_train,y2_train)  
clf_ba3.fit(x3_train,y3_train)
clf_a4.fit(x4_train,y4_train)

# predictions
y1_pred = clf_a1.predict(x1_test)
y2_pred = clf_a2.predict(x2_test)
y3_pred = clf_ba3.predict(x3_test)
y4_pred = clf_a4.predict(x4_test)

# dataset predictions
y1_pred_all = clf_a1.predict(data_best.values)
y2_pred_all = clf_a2.predict(data_best.values)
y3_pred_all = clf_ba3.predict(data_best.values)
y4_pred_all = clf_a4.predict(data_best.values)

# performance
print(80*'-')
print('Test Set Results using Best Features')
print(80*'-')
print()
print(4*' ' + 'Ominous/Tense (music1)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a1.best_params_['clf__C'],clf_a1.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(y1_test, y1_pred)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(y1_test, y1_pred)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(y1_test, y1_pred)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(y1_test, y1_pred)))
print()
print(4*' ' + 'Uplifting (music2)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a2.best_params_['clf__C'],clf_a2.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(y2_test, y2_pred)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(y2_test, y2_pred)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(y2_test, y2_pred)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(y2_test, y2_pred)))
print()
print(4*' ' + 'Sad/Sorrowful (music3)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_ba3.best_params_['clf__C'],clf_ba3.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(y3_test, y3_pred)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(y3_test, y3_pred)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(y3_test, y3_pred)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(y3_test, y3_pred)))
print()
print(4*' ' + 'Ominous/Tense + Sad/Sorrowful (music1 | music3)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a4.best_params_['clf__C'],clf_a4.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(y4_test, y4_pred)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(y4_test, y4_pred)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(y4_test, y4_pred)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(y4_test, y4_pred)))
print()

# dataset performance
print(80*'-')
print('Overall Results using Best Features')
print(80*'-')
print()
print(4*' ' + 'Ominous/Tense (music1)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a1.best_params_['clf__C'],clf_a1.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(music.music1, y1_pred_all)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(music.music1, y1_pred_all)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(music.music1, y1_pred_all)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(music.music1, y1_pred_all)))
print()
print(4*' ' + 'Uplifting (music2)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a2.best_params_['clf__C'],clf_a2.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(music.music2, y2_pred_all)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(music.music2, y2_pred_all)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(music.music2, y2_pred_all)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(music.music2, y2_pred_all)))
print()
print(4*' ' + 'Sad/Sorrowful (music3)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_ba3.best_params_['clf__C'],clf_ba3.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(music.music3, y3_pred_all)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(music.music3, y3_pred_all)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(music.music3, y3_pred_all)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(music.music3, y3_pred_all)))
print()
print(4*' ' + 'Ominous/Tense + Sad/Sorrowful (music1 | music3)')
print(8*' ' + 'Best Params: C = %.1e, gamma = %.1e' 
      % (clf_a4.best_params_['clf__C'],clf_a4.best_params_['clf__gamma']))
print()
print(8*' ' + 'Precision: %0.4f' % (precision_score(np.maximum(music.music1,
                                                                music.music3),
                                                    y4_pred_all)))
print(8*' ' + '   Recall: %0.4f' % (recall_score(np.maximum(music.music1,
                                                                music.music3),
                                                 y4_pred_all)))
print(8*' ' + ' F1-Score: %0.4f' % (f1_score(np.maximum(music.music1,
                                                                music.music3),
                                             y4_pred_all)))
print(8*' ' + ' Accuracy: %0.4f' % (accuracy_score(np.maximum(music.music1,
                                                                music.music3),
                                                   y4_pred_all)))
print()
