import numpy as np
import argparse
import os
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat-path',
                        help='Path to file containing audio features, class '
                             'labels, and audio IDs')
    parser.add_argument('-cc','--combine-class',action='store_true',default=False,
                        help='Combine sad/sorrowful class label with '
                             'ominous/tense class label')

    return parser.parse_args()

def main():
    # read in arguments
    args = parse_arguments()
    feat_path,cc = args.feat_path, args.combine_class
    
    # open file
    feats = pd.read_csv(feat_path)
    