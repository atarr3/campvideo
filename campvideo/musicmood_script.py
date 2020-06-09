import numpy as np
import pandas as pd
import warnings

from os.path import join
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

FEAT_PATH = r'E:\Users\Alex\Desktop\ITH_Matched'
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_english_nodup.csv'
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
data_all = pd.read_csv(join(FEAT_PATH,'mfeats_all.csv'),index_col=0)[music_ind]
data_best = pd.read_csv(join(FEAT_PATH,'mfeats_best.csv'),index_col=0)[music_ind]

####################################
## performance using all features ##
####################################

# train/test splits
x1_train,x1_test,y1_train,y1_test = train_test_split(data_all.values,
                                                      music.music1,
                                                      test_size=0.2,
                                                      random_state=SEED)
x2_train,x2_test,y2_train,y2_test = train_test_split(data_all.values,
                                                      music.music2,
                                                      test_size=0.2,
                                                      random_state=SEED)
x3_train,x3_test,y3_train,y3_test = train_test_split(data_all.values,
                                                      music.music3,
                                                      test_size=0.2,
                                                      random_state=SEED)
# combining music1 with music3
x4_train,x4_test,y4_train,y4_test = train_test_split(data_all.values,
                                                      np.maximum(music.music1,
                                                                music.music3),
                                                      test_size=0.2,
                                                      random_state=SEED)

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
params = [{'clf__gamma': np.logspace(-5,-2,16),
           'clf__C': np.arange(1,10.5,0.5)}]

# fit classifiers

clf_a1 = GridSearchCV(pipe1,params,scoring='accuracy',cv=5)
clf_a2 = GridSearchCV(pipe2,params,scoring='accuracy',cv=5)
clf_ba3 = GridSearchCV(pipe3,params,scoring='balanced_accuracy',cv=5)
clf_a4 = GridSearchCV(pipe4,params,scoring='accuracy',cv=5)

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
y1_pred_all = clf_a1.predict(data_all.values)
y2_pred_all = clf_a2.predict(data_all.values)
y3_pred_all = clf_ba3.predict(data_all.values)
y4_pred_all = clf_a4.predict(data_all.values)


# performance
print(80*'-')
print('Test Set Results using All Features')
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
print('Overall Results using All Features')
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

#####################################
## performance using best features ##
#####################################

# train/test splits
x1_train,x1_test,y1_train,y1_test = train_test_split(data_best.values,
                                                     music.music1,
                                                     test_size=0.2,
                                                     random_state=SEED)
x2_train,x2_test,y2_train,y2_test = train_test_split(data_best.values,
                                                     music.music2,
                                                     test_size=0.2,
                                                     random_state=SEED)
x3_train,x3_test,y3_train,y3_test = train_test_split(data_best.values,
                                                     music.music3,
                                                     test_size=0.2,
                                                     random_state=SEED)
# combining music1 with music3
x4_train,x4_test,y4_train,y4_test = train_test_split(data_best.values,
                                                     np.maximum(music.music1,
                                                                music.music3),
                                                     test_size=0.2,
                                                     random_state=SEED)

# fit classifiers
clf_a1 = GridSearchCV(pipe1,params,scoring='accuracy',cv=5)
clf_a2 = GridSearchCV(pipe2,params,scoring='accuracy',cv=5)
clf_ba3 = GridSearchCV(pipe3,params,scoring='balanced_accuracy',cv=5)
clf_a4 = GridSearchCV(pipe4,params,scoring='accuracy',cv=5)

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
