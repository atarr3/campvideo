import pandas as pd
import pickle
import re
import spacy
import warnings

from campvideo import Video
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
SENT_PATH = resource_filename('campvideo','models/sentiment.pkl')

# issue vocabulary list
VOCAB_PATH = resource_filename('campvideo','data/issuenames.csv')
VOCAB = pd.read_csv(VOCAB_PATH)

# spacy NLP parser
NLP = spacy.load('en_core_web_md')

# regular expression for removing apostrophe's, but preserving those in names
P = re.compile("([\w]{2})'.*$")

# function for checking if any name in NAMES is contained in input string
def has_name(text):
    return any([name in text for name in NAMES])

# tokenizer function
def tokenize(text,ner=True,keep_names=True,keep_pron=False):
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
with open(SENT_PATH,'rb') as fh:
    sent = pickle.load(fh)

# transcript class
class Transcript(object):
    """Video transcript class with methods for sentiment analysis, issue
    mention detection, and candidate/opponent mention detections.

    Parameters
    ----------
    transcript : str
        The raw transcript corresponding to the video.

    parsed : spacy Doc
        The parsed transcript according to spacy.

    cand : str, optional
        The last name of the candidate for the video from which the transcript
        was extracted. The default is None.

    opp : str, optional
        The last name of the candidate for the video from which the transcript
        was extracted. The default is None.

    Attributes
    ----------
    transcript: str
        The raw transcript used to construct the class.
    """

    # constructor
    def __init__(self, transcript, cand=None, opp=None):
        if type(transcript) != str: raise ValueError('invalid input')
        self.transcript = transcript
        self.parsed = NLP(transcript)
        self.cand = cand
        self.opp = opp

    # constructor for when input is video file
    @classmethod
    def fromvid(cls, vid_path, cand=None, opp=None):
        """Construct a Transcript object from a video.

        Parameters
        ----------
        vid_path : str
            The path to the video file.

        cand : str, optional
            The last name of the candidate for the video from which the
            transcript was extracted. The default is None.

        opp : str, optional
            The last name of the candidate for the video from which the
            transcript was extracted. The default is None.

        Returns
        -------
        out : Transcript
            A Transcript object containing the corresponding transcript for the
            input video.
        """
        # construct Video object
        vid = Video(vid_path)

        # get transcript
        transcript = vid.transcribe()

        return cls(transcript, cand=cand, opp=opp)

    # method for computing the sentiment using a pretrained model
    def sentiment(self):
        """Predict the sentiment of the transcript using a pretrained model.

        Returns
        -------
        sentiment : int
            The predicted sentiment of the transcript. 1 if the sentiment is
            positive, 0 if the sentiment is negative.
        """
        # transcript
        transcript = self.transcript

        # sentiment
        sentiment = sent.predict([transcript])[0]

        return sentiment

    # method for detecting opponent mention
    def opp_mention(self, opp=None):
        """Check whether or not the opponent is mentioned in the transcript.

        Parameters
        ----------
        opp : str, optional
            The last name of the opponent for the video from which the
            transcript was extracted. The default is the name passed with the
            constructor. If no name was specified, the function returns None.

        Returns
        -------
        oppment : bool
            True if the opponent's name is contained in the transcript, False
            if it is not.
        """
        # return None if no opponent name available
        if opp is None and self.opp is None: return None
        # set opponent variable
        opp = opp if opp is not None else self.opp
        # get lemmatized tokens
        tokens = [token.lemma_.lower() for token in self.parsed]

        # check for opponent mention
        oppment = opp.lower() in tokens

        return oppment

    # method for detecting candidate mention
    def cand_mention(self, cand=None):
        """Check whether or not the candidate is mentioned in the transcript.

        Parameters
        ----------
        cand : str, optional
            The last name of the candidate for the video from which the
            transcript was extracted. The default is the name passed with the
            constructor. If no name was specified, the function returns None.

        Returns
        -------
        candment : bool
            True if the candidate's name is contained in the transcript, False
            if it is not.
        """
        # return None if no opponent name available
        if cand is None and self.cand is None: return None
        # set opponent variable
        cand = cand if cand is not None else self.cand
        # get lemmatized tokens
        tokens = [token.lemma_.lower() for token in self.parsed]

        # check for opponent mention
        candment = cand.lower() in tokens

        return candment

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

        keep = VOCAB

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
        out = pd.Series(index=keep.yt, dtype=int)
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
