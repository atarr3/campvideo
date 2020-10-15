import argparse
import os
import re
import pandas as pd
import spacy

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


# tokenizer function
def tokenize(text, ner=True, keep_names=True, keep_pron=False):
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

    else:  # ignore named entities
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

from campvideo import Keyframes
from campvideo import Video
from campvideo import Transcript
from timeit import default_timer
from os.path import basename, exists, join, splitext
from pkg_resources import resource_filename
from string import punctuation

# sentiment analysis model
SENT_PATH = resource_filename('campvideo', 'models/sentiment.pkl')

# issue vocabulary list
VOCAB_PATH = resource_filename('campvideo', 'data/issuenames.csv')
VOCAB = pd.read_csv(VOCAB_PATH)

# spacy NLP parser
NLP = spacy.load('en_core_web_md')  # needed to manually download this (dom)

# regular expression for removing apostrophes, but preserving those in names
P = re.compile("([\w]{2})'.*$")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir', metavar='vid-dir',
                        help='Path to video file directory for transcription')
    parser.add_argument('-up', '--use-punct', action='store_true', default=False,
                        help='Enables punctuation annotation for the transcript')
    parser.add_argument('-in', '--incl-names', action='store_true', default=False,
                        help='Includes names in analysis of transcript')
    parser.add_argument('-ip', '--incl-phrases', action='store_true', default=False,
                        help='Includes phrases in analysis of transcript')

    return parser.parse_args()


def main():
    # get CL arguments
    args = parse_arguments()
    vid_dir = args.vid_dir
    use_punct = args.use_punct
    include_names = args.incl_names
    include_phrases = args.incl_phrases

    # load candidate metadata NEED TO DECIDE DIRECTORY
    cands = pd.read_csv('../data/president_metadata.csv')

    # get video paths
    fpaths = [join(root, fname) for root, fold, fnames in os.walk(vid_dir)
              for fname in fnames
              if fname.endswith('.mp4')]
    n_vids = len(fpaths)

    # remove entries for names and phrases if specified
    issues = VOCAB
    if not include_names:
        issues = issues.loc[(issues.cat != 'figure') |
                        (issues.desc == 'Democrats') |
                        (issues.desc == 'Republicans')]
    if not include_phrases:
        issues = issues.loc[issues.cat != 'word']

    # initialize container for data
    issues = issues.yt.to_list()
    issues.extend(['sentiment', 'opp_mention', 'cand_mention', 'transcript',
                   'index', 'fc_lname', 'oc_lname', 'fc_fullname', 'oc_fullname', 'oc_party', 'fc_party', 'office',
                   'election_year', 'incumbent']) # for colnames
    data = []

    # debug file
    with open(join(vid_dir, 'transcription_log.txt'), 'w') as lf:
        # transcribe
        for i, fpath in enumerate(fpaths):
            print('Processing video %d of %d... ' % (i+1, n_vids), end='', flush=True)
            s = default_timer()

            # video name
            cur_name = splitext(basename(fpath))[0]

            # transcribe video
            try:
                # get cand (featured candidate) and opp (opponent candidate) attributes
                opp_lname = cands.loc[cands['file_name'] == cur_name, 'lname_p2'].item()
                cand_lname = cands.loc[cands['file_name'] == cur_name, 'lname'].item()
                opp_nickname = cands.loc[cands['file_name'] == cur_name, 'nickname_2'].item()
                cand_nickname = cands.loc[cands['file_name'] == cur_name, 'nickname_1'].item()
                opp_fullname = cands.loc[cands['file_name'] == cur_name, 'President_2'].item()
                cand_fullname = cands.loc[cands['file_name'] == cur_name, 'President_1'].item()
                opp_party = cands.loc[cands['file_name'] == cur_name, 'Party_2'].item()
                cand_party = cands.loc[cands['file_name'] == cur_name, 'Party_1'].item()
                office = cands.loc[cands['file_name'] == cur_name, 'office_c'].item()
                elect_year = cands.loc[cands['file_name'] == cur_name, 'election_year'].item()
                incumbent = cands.loc[cands['file_name'] == cur_name, 'incumbent'].item()

                # get video object
                vid = Video(fpath)
                print(1)

                # make Keyframe object
                kf = Keyframes.fromvid(fpath)
                print(3)

                # get image text
                texts = kf.keyframes_text()
                print(4)
                # remove empty lists
                texts = [item for item in texts if item != []]
                print(5)

                # transcribe video
                t_str = vid.transcribe()
                print(6)

                # append image text to transcript
                t = t_str + "IMTEXT: " + ' '.join(map(str, texts))
                print(7)

                # create Transcript object for content analysis
                t = Transcript(t)
                print(8)

                # detect cand and opp mentions from transcript
                opp_ment = t.opp_mention(opp_lname) | t.opp_mention(opp_nickname)
                cand_ment = t.cand_mention(cand_lname) | t.cand_mention(cand_nickname)
                soc = {'sentiment': t.sentiment(), 'opp_mention': opp_ment, 'cand_mention': cand_ment,
                       'transcript': t.transcript, 'index': cur_name, 'fc_lname': cand_lname, 'oc_lname': opp_lname,
                       'fc_fullname': cand_fullname, 'oc_fullname': opp_fullname,
                       'fc_party': cand_party, 'oc_party': opp_party, 'office': office, 'election_year': elect_year,
                       'incumbent': incumbent}
                # detect issue mentions
                row = t.issue_mention(include_phrases = include_phrases, include_names = include_names)
                row = row.to_dict()
                row.update(soc)
                data.append(row)
            except Exception as e:
                msg = 'Failed on video `%s` with error: `%s`' % (fpath, str(e))
                print(msg, file=lf)
                print('Failed')
                continue
            print('Done in %4.1f seconds!' % (default_timer()-s))

    df = pd.DataFrame(data, columns = issues)
    df.set_index('index', inplace = True)

    df.to_csv('../data/campvideo.csv')


if __name__ == '__main__':
    main()
