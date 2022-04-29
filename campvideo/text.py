import pandas as pd
import pickle
import re
import spacy
import warnings

import campvideo.video as video
from pkg_resources import resource_filename
from string import punctuation

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

# sentiment analysis model
SENT_PATH = resource_filename('campvideo', 'models/sentiment.pkl')

# issue vocabulary list
_VOCAB_PATH = resource_filename('campvideo', 'data/issuenames.csv')
VOCAB = pd.read_csv(_VOCAB_PATH)

# spacy NLP parser, initially unloaded
NLP = None

# tokenizer, initially unloaded
SENT = None

# regular expression for removing apostrophes, but preserving those in names
P = re.compile("([\w]{2})'.*$")

# custom unpickler for complicated tokenize serialization issue
class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "campvideo.text"
        return super().find_class(module, name)

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
        ents = [P.sub("\\1", ent.lower_) if ent.label_ not in ENTS
                                        else '-' + ent.label_ + '-'
                                    for ent in ents]

        # remove apostrophe suffixes from names
        names = [P.sub("\\1", name.lower_) for name in names]

        # remove names exluding major political figures
        if not keep_names:
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

# load sentiment model (must be done after tokenizer is defined)
with open(SENT_PATH, 'rb') as fh:
    unpickler = MyCustomUnpickler(fh)
    SENT = unpickler.load()
    
# function for creating list of names to check in name mentions
def namegen(name, return_plurals=True):
    # name is a single string for the full name, no punctuation removed
    
    # return None if nan passed
    if type(name) is not str: return [None]
    
    # split names, remove first name
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
    
# transcript class
class Text(object):
    """Video text class with methods for sentiment analysis, issue
    mention detection, and candidate/opponent mention detections.

    Parameters
    ----------
    tpath : str
        The .txt file path to the transcript.

    fav : str, optional
        The full name of the favored candidate for the video from which the 
        transcript was extracted. The default is None.

    opp : str, optional
        The full name of the opposing candidate for the video from which the 
        transcript was extracted. The default is None.

    Attributes
    ----------
    transcript: str
        The raw transcript used to construct the class.
        
    parsed : spacy Doc
        The parsed transcript according to spacy.
    """

    # constructor
    def __init__(self, tpath, fav=None, opp=None):
        if tpath.endswith('.txt'):     
            self.tpath = tpath
            with open(tpath, 'r', encoding='utf8') as fh:
                self.transcript = fh.read()
        elif type(tpath) != str: 
            raise ValueError('invalid input')
        else:
            self.transcript = tpath
        
        # load in spacy parser if not already loaded    
        global NLP
        if NLP is None:
            NLP = spacy.load('en_core_web_md')
        
        self.parsed = NLP(self.transcript)
        self.fav = fav
        self.opp = opp

    # constructor for when input is video file
    @classmethod
    def fromvid(cls, vid_path, fav=None, opp=None):
        """Construct a Transcript object from a video.

        Parameters
        ----------
        vid_path : str
            The path to the video file.

        fav : str, optional
            The last name of the favored candidate for the video from which the
            transcript was extracted. The default is None.

        opp : str, optional
            The last name of the opposing candidate for the video from which 
            the transcript was extracted. The default is None.

        Returns
        -------
        out : Transcript
            A Transcript object containing the corresponding transcript for the
            input video.
        """
        # construct Video object
        vid = video.Video(vid_path)

        # get transcript
        transcript = vid.transcribe()

        return cls(transcript, fav=fav, opp=opp)

    # method for computing the sentiment using a pretrained model
    def sentiment(self):
        """Predict the sentiment of the transcript using a pretrained model.

        Returns
        -------
        sentiment : int
            The predicted sentiment of the transcript. 1 if the sentiment is
            positive, 0 if the sentiment is negative.
        """

        # sentiment
        sentiment = SENT.predict([self.tpath])

        return sentiment

    # method for detecting opponent mention
    def opp_mention(self, opp=None):
        """Check whether or not the opponent is mentioned in the transcript.

        Parameters
        ----------
        opp : str, optional
            The full name of the opponent for the video from which the
            transcript was extracted. The default is the name passed with the
            constructor. If no name was specified, the function returns False.

        Returns
        -------
        opp_ment : bool
            True if the opponent's name is contained in the transcript, False
            if it is not.
        """
        # return False if no opponent name available
        if opp is None and self.opp is None: return False
        # set opponent variable
        opp = opp if opp is not None else self.opp
        # get tokens (no lemmatization)
        tokens = [token.lower_ for token in self.parsed]
        
        # construct list of names to search
        names = namegen(opp)

        # check for opponent mention
        opp_ment = any([name.lower() in tokens for name in names])

        return opp_ment

    # method for detecting candidate mention
    def fav_mention(self, fav=None):
        """Check whether or not the favored candidate is mentioned in the 
        transcript.

        Parameters
        ----------
        fav : str, optional
            The full name of the favored candidate for the video from which the
            transcript was extracted. The default is the name passed with the
            constructor. If no name was specified, the function returns None.

        Returns
        -------
        fav_ment : bool
            True if the candidate's name is contained in the transcript, False
            if it is not.
        """
        # return None if no candidate name available
        if fav is None and self.fav is None: return None
        # set candidate variable
        fav = fav if fav is not None else self.fav
        # get lemmatized tokens
        tokens = [token.lower_ for token in self.parsed]
        
        # construct list of names to search
        names = namegen(fav)

        # check for candidate mention
        fav_ment = any([name.lower() in tokens for name in names])

        return fav_ment

    # method for detecting issue mentions
    def issue_mention(self, include_names=False, include_phrases=False):
        """Check whether or not specific issues are mentioned in the
        transcript. See the paper for the corresponding list of issues and
        which words were used to detect issue mentions.

        Parameters
        ----------
        include_names : bool, optional
            Boolean flag specifying whether or not to detect the mention of
            prominent political figures during the period of 2012-2016. These
            names include Barack Obama, George W. Bush, Ronald Reagan, John
            Boehner, Nancy Pelosi, Mitch McConnell, and Harry Reid. The default
            is False.

        include_phrases : bool, optional
            Boolean flag specifying whether or not to detect politically
            charged words or phrases during the period 2012-2016. These phrases
            include Tea party, God, hope, change, experience, Liberal,
            Conservative, special interest, negative campaigning, Main Street,
            Wall Street, and big government. The default is False.

        Returns
        -------
        out : pandas Series
            A named array of boolean values specifying whether or not the
            issue was mentioned in the transcript. A 1 corresponds to an issue
            mention, while a 0 corresponds to an issue not being mentioned.
        """
        # get transcript
        transcript = self.transcript.lower()

        keep = VOCAB.copy()

        # remove entries for names and phrases if specified
        if not include_names:
            keep = keep.loc[(keep.cat != 'figure') |
                            (keep.desc == 'Democrats') |
                            (keep.desc == 'Republicans')]
        if not include_phrases:
            keep = keep.loc[keep.cat != 'word']
        # reset index
        keep.reset_index(drop=True, inplace=True)

        # iterate through rows and check issue mentions
        out = pd.Series(0, index=keep.wmp, dtype=int)
        for i, desc, _, _, _, _, word, noword in keep.itertuples():
            # word count (words that correspond to an issue)
            words = word.split('|')
            wc = sum(transcript.count(w) for w in words)

            # noword count (exceptions to the rule for detection)
            nowords = noword.split('|') if not pd.isnull(noword) else []
            nwc = sum(transcript.count(nw) for nw in nowords)

            # update output
            out.iloc[i] = 1 if wc > nwc else 0

        return out
