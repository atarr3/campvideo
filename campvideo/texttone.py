import os
import numpy as np
import pandas as pd
import re
import spacy
import warnings

from os.path import join
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from sklearn.metrics import classification_report as cr, confusion_matrix as cm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from string import punctuation

TEXT_PATH = r'E:\Users\Alex\Desktop\ITH_Matched\transcripts'
MUSIC_PATH = r'E:\Users\Alex\Desktop\ITH_Matched\mfeats_best.csv'
OUT_PATH = r'E:\Users\Alex\Desktop'
WMP_PATH = r'E:\Users\Alex\Desktop\wmp_final.csv'
SEED = 2002

# annoying sklearn warning suppression
warnings.filterwarnings('ignore') 

# Modified list of stop words taken from
# https://www.ranks.nl/stopwords
STOP = frozenset([
    "approve","message",
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", 
    "been",  "being", "both", "by",  "during", "each", "for", "from", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "let's", "me",  "must", "my", "myself", "nor",  "of", "once", "or", "other", 
    "ought", "ourselves", "own", "shall", "she", "should", "so", "some", "such", 
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "to", "until", "up", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "would", 
    "you", "your", "yours", "yourself", "yourselves", "'s"])

# list of generic entity descriptors
ENTS = frozenset(["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
                  "CARDINAL"])

# list of names to keep when removing names
NAMES = frozenset(["barack", "obama", "barack obama", "pelosi", "nancy pelosi",
                   "reagan", "ronald reagan"])

# spacy NLP parser
NLP = spacy.load('en_core_web_md')

# regular expression for removing apostrophes, but preserving those in names
P = re.compile("([\w]{2})'.*$")

# function for checking if any name in NAMES is contained in input string
def has_name(text):
    return any([name in text for name in NAMES])

# tokenizer function
def tokenize(text, ner=True, keep_names=False, keep_pron=False):
    # parse text using spacy
    parsed = NLP(text)
    
    # split named entities
    if ner:
        # lemmatize non-entity tokens
        tokens = [token.lemma_ for token in parsed 
                                   if token.ent_type == 0]
        # convert to lowercase, excluding pronouns
        tokens = [token.lower() if token != "-PRON-" else token 
                            for token in tokens]
        
        # split names from rest of entities
        names = [ent for ent in parsed.ents if ent.label_ == "PERSON"]
        ents = [ent for ent in parsed.ents if ent.label_ != "PERSON"]
        
        # convert numeric entities to generic description and remove apostrophe
        # suffixes
        ents = [P.sub("\\1",ent.lower_) if ent.label_ not in ENTS 
                                        else '-' + ent.label_ + '-'
                                    for ent in ents]
        
        # remove apostrophe suffixes from names
        names = [P.sub("\\1",name.lower_) for name in names]
        
        # relabel names to -NAME- exluding major political figures
        if not keep_names:
            # names = [name if has_name(name) else '-NAME-' for name in names]
            names = [name for name in names if has_name(name)]   
    else: # ignore named entities
        # lemmatize all tokens
        tokens = [token.lemma_.lower() for token in parsed]
        
                  
    # keep pronouns and remove all other stop words / punctuation
    if keep_pron:
        tokens = [token for token in tokens
                            if token not in STOP and token not in punctuation]
    else:
        tokens = [token for token in tokens
                            if token not in STOP and token not in punctuation
                            and token != '-PRON-']
    
    # return list of tokens
    if ner:
        return tokens + ents + names
    else:
        return tokens
    
# class implementing naive Bayes for heterogenous data
class HeterogenousNB(BaseEstimator, ClassifierMixin):
    def __init__(self, discrete_clf='bernoulli', alpha=1.0, 
                 fit_prior=True, class_prior=None):
        
        self.discrete_clf = discrete_clf
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y, split_ind=-1):
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # check discrete_clf
        if self.discrete_clf.lower() not in ['bernoulli', 'multinomial']:
            raise ValueError("Incorrect classifier for discrete data, "
                             "please specify 'bernoulli' or 'multinomial'")
            
        # index marking split between discrete (1st segment) and continuous data
        self.split_ind_ = split_ind 
        
        # store the classes seen during fit
        self.classes_ = unique_labels(y)
    
        self.X_ = X
        self.y_ = y
        
        # fit naive Bayes classifiers
        X_disc, X_cont = np.split(X, np.atleast_1d(split_ind), axis=1)
        
        if self.discrete_clf == 'bernoulli':
            nb_disc = BernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                  class_prior=self.class_prior).fit(X_disc, y)
        else:
            nb_disc = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                    class_prior=self.class_prior).fit(X_disc, y) 
        nb_cont = GaussianNB(priors=self.class_prior).fit(X_cont, y)
        
        # save fitted classifiers as attributes
        self.nb_disc_ = nb_disc
        self.nb_cont_ = nb_cont
        
        # return the classifier
        return self

    def predict(self, X):
        # check is fit had been called
        check_is_fitted(self)
    
        # input validation
        X = check_array(X)
        
        # prior probs
        log_prior = self.nb_disc_.class_log_prior_
        
        # GNB params
        theta = self.nb_cont_.theta_
        sigma = self.nb_cont_.sigma_
        
        # BNB / MNB params
        logp = self.nb_disc_.feature_log_prob_
        
        # compute joint log-likelihood
        X_disc, X_cont = np.split(X, np.atleast_1d(self.split_ind_), axis=1)
        
        jll_disc = np.dot(X_disc, logp.T)
        jll_cont = []
        for i in range(np.size(self.classes_)):
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * sigma[i, :]))
            n_ij -= 0.5 * np.sum(((X_cont - theta[i, :]) ** 2) / 
                                 (sigma[i, :]), 1)
            jll_cont.append(n_ij)
            
        jll_cont = np.array(jll_cont).T
        
        # total joint log-likelihood
        jll_total = log_prior + jll_disc + jll_cont
    
        return self.classes_[np.argmax(jll_total, axis=1)]

###################
## Training Data ##
###################
        
# output file
if os.path.exists(join(OUT_PATH,'texttone_results.txt')):
    os.rename(join(OUT_PATH,'texttone_results.txt'), 
              join(OUT_PATH,'texttone_results_old.txt'))
        
# read in wmp data, sort, and remove nan
wmp = pd.read_csv(WMP_PATH, index_col='uid', usecols=['uid','tonecmag']
                 ).sort_index(key=lambda x: x.str.lower()
                 ).dropna(subset=['tonecmag']
                 )
# drop 'contrast' observations
wmp.drop(wmp[wmp.tonecmag == 'CONTRAST'].index, inplace=True)
# recast tonecmag to 1/0 for sentiment
wmp.tonecmag = ((wmp.tonecmag == 'POS') | (wmp.tonecmag == 'POSITIVE')).astype(int)

# get list of transcript paths from wmp file
tpaths = [join(TEXT_PATH, uid + '.txt') for uid in wmp.index] 

# get music features
mfeats = pd.read_csv(MUSIC_PATH, index_col=0
                    ).sort_index(key=lambda x : x.str.lower())
mfeats = mfeats.loc[wmp.index]
d = mfeats.shape[1]

# combine
feats = mfeats.assign(tpath=tpaths)

# train/test splits
x_train,x_test,y_train,y_test = train_test_split(feats,
                                                 wmp.tonecmag,
                                                 test_size=0.2,
                                                 random_state=SEED)

################
## Linear SVM ##
################

## text only ##

# pipeline
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                input='filename'), -1)],
                        remainder='drop')),
            ('dim_red', SelectPercentile(mutual_info_classif)),
            ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'dim_red__percentile' : [50, 75, 90, 100],
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__class_weight': ['balanced', None]
         }]

# grid search
lsvm_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
lsvm_t.fit(x_train, y_train)

# predictions and performance
y_pred_tr = lsvm_t.predict(x_train)
y_pred_te = lsvm_t.predict(x_test)
y_pred_al = lsvm_t.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Linear SVM (Text Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(lsvm_t.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## music only ##

# pipeline    
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("ss", StandardScaler(), slice(-1))],
                        remainder='drop')),
            ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__class_weight': ['balanced', None]
         }]

# grid search
lsvm_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
lsvm_m.fit(x_train, y_train)

# predictions and performance
y_pred_tr = lsvm_m.predict(x_train)
y_pred_te = lsvm_m.predict(x_test)
y_pred_al = lsvm_m.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Linear SVM (Music Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(lsvm_m.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## text + music ##
    
# pipeline    
pipe = Pipeline([
            ('feat1', ColumnTransformer(
                            [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                    input='filename'), -1)],
                            remainder='passthrough')),
            # music features run from -d:
            ('feat2', ColumnTransformer(
                        [("dim_red", SelectPercentile(mutual_info_classif), 
                          slice(-d))], remainder='passthrough')),
            # SVM inputs should be standardized
            ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                           accept_sparse=True)),
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'feat2__dim_red__percentile' : [50, 75, 90, 100],
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__class_weight': ['balanced', None]
         }]

# grid search
lsvm_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
lsvm_tm.fit(x_train, y_train)

# predictions and performance
y_pred_tr = lsvm_tm.predict(x_train)
y_pred_te = lsvm_tm.predict(x_test)
y_pred_al = lsvm_tm.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Linear SVM (Text + Music)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(lsvm_tm.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)
    
###################
## Nonlinear SVM ##
###################
    
## text only ##

# pipeline
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                input='filename'), -1)],
                        remainder='drop')),
            ('dim_red', SelectPercentile(mutual_info_classif)),
            ('clf', SVC(kernel='rbf', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'dim_red__percentile' : [50, 75, 90, 100],
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__gamma': np.logspace(-6, -2, 5),
           'clf__class_weight': ['balanced', None]
         }]

# grid search
svm_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
svm_t.fit(x_train, y_train)

# predictions and performance
y_pred_tr = svm_t.predict(x_train)
y_pred_te = svm_t.predict(x_test)
y_pred_al = svm_t.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Nonlinear SVM (Text Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(svm_t.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## music only ##

# pipeline    
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("ss", StandardScaler(), slice(-1))],
                        remainder='drop')),
            ('clf', SVC(kernel='rbf', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__gamma': np.logspace(-6, -2, 5),
           'clf__class_weight': ['balanced', None]
         }]

# grid search
svm_m = GridSearchCV(pipe,params,scoring='accuracy',cv=5,verbose=1)
svm_m.fit(x_train, y_train)

# predictions and performance
y_pred_tr = svm_m.predict(x_train)
y_pred_te = svm_m.predict(x_test)
y_pred_al = svm_m.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Nonlinear SVM (Music Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(svm_m.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## text + music ##
    
# pipeline    
pipe = Pipeline([
            ('feat1', ColumnTransformer(
                            [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                    input='filename'), -1)],
                            remainder='passthrough')),
            # music features run from -d:
            ('feat2', ColumnTransformer(
                        [("dim_red", SelectPercentile(mutual_info_classif), 
                          slice(-d))], remainder='passthrough')),
            # SVM inputs should be standardized
            ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                           accept_sparse=True)),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
           'feat2__dim_red__percentile' : [50, 75, 90, 100],
           'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
           'clf__gamma': np.logspace(-6, -2, 5),
           'clf__class_weight': ['balanced', None]
         }]

# grid search
svm_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
svm_tm.fit(x_train, y_train)

# predictions and performance
y_pred_tr = svm_tm.predict(x_train)
y_pred_te = svm_tm.predict(x_test)
y_pred_al = svm_tm.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Nonlinear SVM (Text + Music)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(svm_tm.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)
    
#########
## KNN ##
#########
    
## text only ##

# pipeline
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                input='filename'), -1)],
                        remainder='drop')),
            ('dim_red', SelectPercentile(mutual_info_classif)),
            ('clf', KNeighborsClassifier())
                ])

# parameter grid for grid search
params = [{
          'dim_red__percentile' : [50, 75, 90, 100],
          'clf__n_neighbors': [5, 7, 11, 15, 21]
         }]

# grid search
knn_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
knn_t.fit(x_train, y_train)

# predictions and performance
y_pred_tr = knn_t.predict(x_train)
y_pred_te = knn_t.predict(x_test)
y_pred_al = knn_t.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("KNN (Text Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(knn_t.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## music only ##

# pipeline    
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("ss", StandardScaler(), slice(-1))],
                        remainder='drop')),
            ('clf', KNeighborsClassifier())
                ])

# parameter grid for grid search
params = [{
          'clf__n_neighbors': [5, 7, 11, 15, 21]
         }]

# grid search
knn_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
knn_m.fit(x_train, y_train)

# predictions and performance
y_pred_tr = knn_m.predict(x_train)
y_pred_te = knn_m.predict(x_test)
y_pred_al = knn_m.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("KNN (Music Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(knn_m.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## text + music ##
    
# pipeline    
pipe = Pipeline([
            ('feat1', ColumnTransformer(
                            [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                    input='filename'), -1)],
                            remainder='passthrough')),
            # music features run from -d:
            ('feat2', ColumnTransformer(
                        [("dim_red", SelectPercentile(mutual_info_classif), 
                          slice(-d))], remainder='passthrough')),
            # KNN inputs should be standardized
            ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                           accept_sparse=True)),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
                ])

# parameter grid for grid search
params = [{
          'feat2__dim_red__percentile' : [50, 75, 90, 100],
          'clf__n_neighbors': [5, 7, 11, 15, 21]
         }]

# grid search
knn_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
knn_tm.fit(x_train, y_train)

# predictions and performance
y_pred_tr = knn_tm.predict(x_train)
y_pred_te = knn_tm.predict(x_test)
y_pred_al = knn_tm.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("KNN (Text + Music)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(knn_tm.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)
    
###################
## Random Forest ##
###################
    
## text only ##

# pipeline
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                input='filename'), -1)],
                        remainder='drop')),
            ('dim_red', SelectPercentile(mutual_info_classif)),
            ('clf', RandomForestClassifier(class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
          'dim_red__percentile' : [50, 75, 90, 100],
          'clf__n_estimators': [100, 250, 500],
          'clf__min_samples_leaf': [1, 2, 4],
          'clf__min_samples_split': [2, 5, 10],
          'clf__class_weight' : ['balanced', None]
         }]

# grid search
rf_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
rf_t.fit(x_train, y_train)

# predictions and performance
y_pred_tr = rf_t.predict(x_train)
y_pred_te = rf_t.predict(x_test)
y_pred_al = rf_t.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Random Forest (Text Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(rf_t.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## music only ##

# pipeline    
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [('ss', 'passthrough', slice(-1))], 
                        remainder='drop')),
            ('clf', RandomForestClassifier(class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
          'clf__n_estimators': [100, 250, 500],
          'clf__min_samples_leaf': [1, 2, 4],
          'clf__min_samples_split': [2, 5, 10],
          'clf__class_weight' : ['balanced', None]
         }]

# grid search
rf_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
rf_m.fit(x_train, y_train)

# predictions and performance
y_pred_tr = rf_m.predict(x_train)
y_pred_te = rf_m.predict(x_test)
y_pred_al = rf_m.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Random Forest (Music Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(rf_m.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

## text + music ##
    
# pipeline    
pipe = Pipeline([
            ('feat1', ColumnTransformer(
                            [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                    input='filename'), -1)],
                            remainder='passthrough')),
            # music features run from -d:
            ('feat2', ColumnTransformer(
                        [("dim_red", SelectPercentile(mutual_info_classif), 
                          slice(-d))], remainder='passthrough')),
            ('clf', RandomForestClassifier(class_weight='balanced'))
                ])

# parameter grid for grid search
params = [{
          'feat2__dim_red__percentile' : [50, 75, 90, 100],
          'clf__n_estimators': [100, 250, 500],
          'clf__min_samples_leaf': [1, 2, 4],
          'clf__min_samples_split': [2, 5, 10],
          'clf__class_weight' : ['balanced', None]
         }]

# grid search
rf_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
rf_tm.fit(x_train, y_train)

# predictions and performance
y_pred_tr = rf_tm.predict(x_train)
y_pred_te = rf_tm.predict(x_test)
y_pred_al = rf_tm.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Random Forest (Text + Music)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(rf_tm.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

#################    
## Naive Bayes ##
#################
    
## text only ##

# pipeline
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                input='filename'), -1)],
                        remainder='drop')),
            ('dim_red', SelectPercentile(mutual_info_classif)),
            ('clf', MultinomialNB())
                ])

# parameter grid for grid search
params = [{
          'feat__cv' : [CountVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'dim_red__percentile': [50, 75, 90, 100],
          'clf': [BernoulliNB(), MultinomialNB()],
          'clf__alpha': [0.01, 0.1, 1, 2] 
         },
         {
          'feat__cv' : [TfidfVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'dim_red__percentile': [50, 75, 90, 100],
          'clf': [MultinomialNB()],
          'clf__alpha': [0.01, 0.1, 1, 2] 
         },
         {
          'feat__cv' : [TfidfVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'dim_red__percentile': [50, 75, 90, 100],
          'clf': [GaussianNB()]
         }]

# grid search
nb_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
nb_t.fit(x_train, y_train)

# predictions and performance
y_pred_tr = nb_t.predict(x_train)
y_pred_te = nb_t.predict(x_test)
y_pred_al = nb_t.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Naive Bayes (Text Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(nb_t.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)

# music only
    
# pipeline    
pipe = Pipeline([
            ('feat', ColumnTransformer(
                        [('ss', 'passthrough', slice(-1))], 
                        remainder='drop')),
            ('clf', GaussianNB())
                ])

# fit classifier (no parameters to tune)
nb_m = pipe
nb_m.fit(x_train, y_train)

# predictions and performance
y_pred_tr = nb_m.predict(x_train)
y_pred_te = nb_m.predict(x_test)
y_pred_al = nb_m.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Naive Bayes (Music Only)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)    

## text + music ##
    
# pipeline    
pipe = Pipeline([
            ('feat1', ColumnTransformer(
                            [("cv", TfidfVectorizer(analyzer=tokenize, min_df=2,
                                                    input='filename'), -1)],
                            remainder='passthrough')),
            # music features run from -d:
            ('feat2', ColumnTransformer(
                        [("dim_red", SelectPercentile(mutual_info_classif), 
                          slice(-d))], remainder='passthrough')),
            # make output array dense
            ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                           accept_sparse=True)),
            ('clf', HeterogenousNB())
                ])

# parameter grid for grid search
params = [{
          'feat1__cv' : [CountVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'feat2__dim_red__percentile': [50, 75, 90, 100],
          'clf__discrete_clf': ['bernoulli', 'multinomial'],
          'clf__alpha': [0.01, 0.1, 1, 2] 
         },
         {
          'feat1__cv' : [TfidfVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'feat2__dim_red__percentile': [50, 75, 90, 100],
          'clf__discrete_clf': ['multinomial'],
          'clf__alpha': [0.01, 0.1, 1, 2] 
         },
         {
          'feat1__cv' : [TfidfVectorizer(analyzer=tokenize, min_df=2,
                                        input='filename')],
          'feat2__dim_red__percentile': [50, 75, 90, 100],
          'clf': [GaussianNB()]
         }]

# grid search
nb_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
nb_tm.fit(x_train, y_train, clf__split_ind=-d)

# predictions and performance
y_pred_tr = nb_tm.predict(x_train)
y_pred_te = nb_tm.predict(x_test)
y_pred_al = nb_tm.predict(feats)

with open(join(OUT_PATH,'texttone_results.txt'),'a') as fh:
    print("Naive Bayes (Text + Music)",file=fh)
    print(80*'-'+'\n', file=fh)
    print("Training:",file=fh)
    print(nb_tm.best_params_,file=fh)
    print('',file=fh)
    print(cr(y_train, y_pred_tr, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_train, y_pred_tr),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Test:",file=fh)
    print(cr(y_test, y_pred_te, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(y_test, y_pred_te),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('',file=fh)
    print("Overall:",file=fh)
    print(cr(wmp.tonecmag, y_pred_al, target_names = ['neg','pos']),file=fh)
    print('',file=fh)
    print(pd.DataFrame(cm(wmp.tonecmag, y_pred_al),
                       index=['true:neg','true:pos'],columns=['pred:neg','pred:pos']
                       ),file=fh)
    print('\n',file=fh)   
    