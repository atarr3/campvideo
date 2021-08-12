import pandas as pd
import pickle
import re
import spacy
import warnings

from functools import partial
from os.path import basename, dirname, join, splitext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from string import punctuation

FEAT_PATH = r'E:\Users\Alex\Desktop\ITH_Matched\transcripts'
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

# regular expression for removing apostrophe's, but preserving those in names
P = re.compile("([\w]{2})'.*$")

# function for checking if any name in NAMES is contained in input string
def has_name(text):
    return any([name in text for name in NAMES])

# tokenizer function
def tokenize(text,ner=True,keep_names=True,keep_pron=True):
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
            names = [name if has_name(name) else '-NAME-' for name in names]
               
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
    
##########
## Data ##
##########

if __name__ == "__main__":
        
    # read in wmp data, sort, and remove nan
    wmp = pd.read_csv(WMP_PATH, index_col='uid',
                      usecols=['uid','tonecmag']
                      ).sort_index(
                      ).dropna(subset=['tonecmag']
                      )
    # drop 'contrast' observations
    wmp.drop(wmp[wmp.tonecmag == 'CONTRAST'].index, inplace=True)
    # recast tonecmag to 1/0 for sentiment
    wmp.tonecmag = ((wmp.tonecmag == 'POS') | (wmp.tonecmag == 'POSITIVE')).astype(int)
    
    # get list of transcript paths from wmp file
    tpaths = [join(FEAT_PATH,uid + '.txt') for uid in wmp.index] 
    
    ##############
    ## Training ##
    ##############
    
    # train/test splits
    x_train,x_test,y_train,y_test = train_test_split(tpaths,
                                                     wmp.tonecmag,
                                                     test_size=0.2,
                                                     random_state=SEED)
            
    # classification pipeline
    pipe = Pipeline([
                ('feat', CountVectorizer(input='filename')),
                ('dim_red', SelectPercentile(chi2)),
                ('clf', MultinomialNB())
                    ])
    
    # parameter grids for grid search
    params = [{'feat': [CountVectorizer(input='filename'), 
                        TfidfVectorizer(input='filename')],
               'feat__analyzer': [partial(tokenize, ner=True, keep_names=True,
                                          keep_pron=True),
                                  partial(tokenize,ner=True, keep_names=True,
                                          keep_pron=False),
                                  partial(tokenize, ner=True, keep_names=False,
                                          keep_pron=True),
                                  partial(tokenize, ner=True, keep_names=False,
                                          keep_pron=False),
                                  partial(tokenize,ner=False, keep_pron=True),
                                  partial(tokenize,ner=False, keep_pron=False)],
               'feat__min_df': [1, 2, 5],
               'dim_red__percentile': [50, 75, 90, 100],
               'clf': [BernoulliNB()],
               'clf__alpha': [0.01, 0.1, 1, 2]}]
    
    # grid search
    clf_a = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=2)
    
    # fit
    clf_a.fit(x_train, y_train)
    
    # save results
    cv_res = pd.DataFrame(clf_a.cv_results_)
    cv_res.to_csv(join(dirname(WMP_PATH), 'cv_results.csv'), index=False)
    
    with open(join(dirname(FEAT_PATH), 'textfeat.pkl'),"wb") as fh:
        pickle.dump(clf_a, fh)
