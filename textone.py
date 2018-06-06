import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import os

from gensim import corpora

from sklearn import preprocessing, svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split


from collections import defaultdict, OrderedDict
from nltk.tag.stanford import StanfordNERTagger
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# cmag data directory
CMAG_DIR = 'C:\\Users\\atarr\\OneDrive\\Documents\\Research\\campvideo\\U2D Documents'
# feature path
FEAT_DIR = 'C:\\Users\\atarr\\OneDrive\\Documents\\Research\\campvideo\\data\\features\\ITH_Video.npy'
# IDs path
IDS_DIR = 'C:\\Users\\atarr\\OneDrive\\Documents\\Research\\campvideo\\data\\features\\video_ids.txt'    

# stemmer and tagger
STOP = set(stopwords.words('english'))
SS = SnowballStemmer('english')
ST = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

# tokenizing function
def tokenize(chunks):
    # initialize list
    output = []
    
    # parse chunked tree
    for c in chunks:
        if type(c) == Tree: # named entity, remove people
            c_ne = ' '.join([token.title() 
                             for token, pos in c.leaves()]) #if pos != 'PERSON'])
            if c_ne:
                output.append(c_ne)
        else:
            syls = [SS.stem(syl) for syl in c[0].lower().split("'") 
                                                            if syl not in STOP]
            for syl in syls:
                output.append(syl)

    return output

def label_ne(text):
    # tag named entities
    tags = ST.tag(text.split())
    
    # chunk by entity
    grammar = r"""
    NE:
        {<PERSON>+}
        {<ORGANIZATION>+}
        {<LOCATION>+}
    """
    cp = nltk.RegexpParser(grammar)
    ne_tree = cp.parse(tags)
    
    return ne_tree

# read in WMP data
cmag12p = pd.read_csv(os.path.join(CMAG_DIR,'matchedtable12p.csv'),
                      usecols=['uid','script','score','wcount','caption','cwcount','tonecmag','ad_tone'])
cmag12 = pd.read_csv(os.path.join(CMAG_DIR,'matchedtable12.csv'),
                     usecols=['uid','script','score','wcount','caption','cwcount','tonecmag','ad_tone'])
cmag14 = pd.read_csv(os.path.join(CMAG_DIR,'matchedtable14.csv'),
                     usecols=['uid','script','score','wcount','caption','cwcount','tonecmag','ad_tone'])
                     
# merge and remove rows with nan
cmag = pd.concat([cmag12,cmag12p,cmag14],ignore_index=True).dropna(axis=0,how='any')

# remove unmatched videos
cmag = cmag[(cmag.uid != 'NoMatch') & (cmag.uid != 'NoChannel')]

# convert numeric data to integers
cmag = cmag.apply(pd.to_numeric,errors='ignore',downcast='integer')

# drop duplicate entries (by uid)
cmag_nodup = cmag.drop_duplicates(subset='uid')

# remove samples with tone CONTRAST
pn = cmag_nodup[cmag_nodup.tonecmag != 'CONTRAST']

# convert tonecmag to 0/1
pn.loc[:,'tonecmag'] = ((pn['tonecmag'] == 'POS') | 
                        (pn['tonecmag'] == 'POSITIVE')).astype(int)
                        
# read in feature data and list of corresponding IDs
feats = np.load(FEAT_DIR)
with open(IDS_DIR,'r') as fn:
    raw = fn.readlines()
    feats_ids = [item.strip() for item in raw]
# subset down to entries in pn
feats = np.array([feats[feats_ids.index(cur_id)] for cur_id in pn.uid])

# unprocessed feature data (pre-text processing)
data_x = pd.DataFrame(feats,index=pn.index)
data_x['caption'] = pn.caption
# tone
data_y = pn.tonecmag

# split into testing and training
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,
                                                 random_state=250)
   
# # build dictionaries from training data
# # word counts
# dict_p = defaultdict(int)
# dict_n = defaultdict(int)
# 
# # positive ads
# tp = len(x_train.caption[y_train == 1])
# for i,text in enumerate(x_train.caption[y_train == 1]):
#     print('tokenizing positive file {0} of {1}'.format(i+1,tp))
#     for token in tokenize(label_ne(text)):
#         dict_p[token] += 1
# 
# # negative ads
# tn = len(x_train.caption[y_train == 0])
# for i,text in enumerate(x_train.caption[y_train == 0]):
#     print('tokenizing negative file {0} of {1}'.format(i+1,tn))
#     for token in tokenize(label_ne(text)):
#         dict_n[token] += 1
# 
# # sort by frequency        
# words_p = OrderedDict(sorted(dict_p.items(),key=lambda t: t[1],reverse=True))
# words_n = OrderedDict(sorted(dict_n.items(),key=lambda t: t[1],reverse=True))
# 
# # save dictionaries
# with open('positive.pkl','wb') as f:
#     pickle.dump(words_p,f)
#     
# with open('negative.pkl','wb') as f:
#     pickle.dump(words_n,f)
    
# pos and neg vocabulary
with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f) 
    
# bag of words for full vocabulary
tokens_full = [tokenize(label_ne(text)) for text in x_train.caption]
# bag of words for pos/neg vocabulary
tokens_pn = [list(filter(lambda item: item in vocab, token)) 
             for token in tokens_full]

# full and partial text
text_full = [' '.join(item) for item in tokens_full]
text_pn = [' '.join(item) for item in tokens_pn]

# full and partial vocabulary corpora                 
d_full = corpora.Dictionary(tokens_full)
d_pn = corpora.Dictionary(tokens_pn)

# counts transformer
vec_pn = CountVectorizer(lowercase=False,vocabulary=d_pn.token2id)
vec_full = CountVectorizer(lowercase=False,vocabulary=d_full.token2id)

counts_pn = vec_pn.transform(text_pn).toarray()
counts_full = vec_full.transform(text_full).toarray()

# TF-IDF transformer
transformer = TfidfTransformer()

tfidf_full = transformer.fit_transform(counts_full).toarray()

# build feature data for training SVM
feats_full_train = np.hstack((x_train.iloc[:,:-1].as_matrix(),tfidf_full))
feats_pn_train = np.hstack((x_train.iloc[:,:-1].as_matrix(),counts_pn))

# build feature data for testing SVM
tokens_full_test = [tokenize(label_ne(text)) for text in x_test.caption]
# bag of words for pos/neg vocabulary
tokens_pn_test = [list(filter(lambda item: item in vocab, token)) 
                  for token in tokens_full_test]
                  
# full and partial text
text_full_t = [' '.join(item) for item in tokens_full_test]
text_pn_t = [' '.join(item) for item in tokens_pn_test]
counts_pn_t = vec_pn.transform(text_pn_t).toarray()
counts_full_t = vec_full.transform(text_full_t).toarray()
tfidf_full_t = transformer.fit_transform(counts_full_t).toarray()

feats_full_test = np.hstack((x_test.iloc[:,:-1].as_matrix(),tfidf_full_t))
feats_pn_test = np.hstack((x_test.iloc[:,:-1].as_matrix(),counts_pn_t))
                                                       
# compute hyperparameters using grid search
params = [{'gamma': 2.0 ** np.arange(-9,-6,0.5),
           'C': np.arange(1.2,1.7,0.10)}]
svc = svm.SVC(kernel='rbf',class_weight='balanced')

# music only
s_m = preprocessing.StandardScaler().fit(feats_full_train[:,:636])
clf_m = GridSearchCV(svc,params,scoring='f1',cv=5)
clf_m.fit(s_m.transform(feats_full_train[:,:636]),y_train)
y_pred_m = clf_m.predict(s_m.transform(feats_full_test[:,:636]))

# text only (full)
s_tf = preprocessing.StandardScaler().fit(feats_full_train[:,636:])
clf_tf = GridSearchCV(svc,params,scoring='f1',cv=5)
clf_tf.fit(s_tf.transform(feats_full_train[:,636:]),y_train)
y_pred_tf = clf_tf.predict(s_tf.transform(feats_full_test[:,636:]))

# text only (partial)
s_tp = preprocessing.StandardScaler().fit(feats_pn_train[:,636:])
clf_tp = GridSearchCV(svc,params,scoring='f1',cv=5)
clf_tp.fit(s_tp.transform(feats_pn_train[:,636:]),y_train)
y_pred_tp = clf_tp.predict(s_tp.transform(feats_pn_test[:,636:]))

# music + partial
s_mp = preprocessing.StandardScaler().fit(feats_pn_train)
clf_mp = GridSearchCV(svc,params,scoring='f1',cv=5)
clf_mp.fit(s_mp.transform(feats_pn_train),y_train)
y_pred_mp = clf_mp.predict(s_mp.transform(feats_pn_test))

# music + full
s_mf = preprocessing.StandardScaler().fit(feats_full_train)
clf_mf = GridSearchCV(svc,params,scoring='f1',cv=5)
clf_mf.fit(s_mf.transform(feats_full_train),y_train)
y_pred_mf = clf_mf.predict(s_mf.transform(feats_full_test))