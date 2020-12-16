import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import os 
# import sys
# sys.path.append('../Code')
# sys.path.append('../Data')

def variance_threshold(X_train, cutoff=0.8):
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=(cutoff*(1-cutoff)))
    selector.fit_transform(X_train)

    return selector.get_support()
    # return selector


def feature_selection(X_train,y_train,method='chi2',plot=False, filename=None):
    '''
    Performs feature selection on input data using given method. Supported 
    choices include 'chi2', 'mutual_info' and 'f_classif' (ANOVA f-test). 
    If plot=True a barplot of the scoring is plotted. If a filename is given
    the plot is saved with this name. Otherwose it is displayed.
    '''
    from sklearn.feature_selection import SelectKBest

    if(method=='chi2'):
        from sklearn.feature_selection import chi2
        selector = SelectKBest(chi2, k='all')
    elif(method=='mutual_info'):
        from sklearn.feature_selection import mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k='all')
    elif(method=='f_classif'):
        from sklearn.feature_selection import f_classif
        selector = SelectKBest(f_classif, k='all')
    else:
        raise Exception(f"Chosen method of feature selection \'{method}\' is not supported.")
        return
    
    selector.fit(X_train, y_train)
    scores = selector.scores_

    if plot:
        scores /= scores.max()
        fig, ax = plt.subplots(figsize =(12, 8)) 
        plt.bar(np.arange(X_train.shape[-1]), scores, width=.2)
        plt.title(f"The relative {method} scores for each feature")
        plt.xlabel("feature index")
        plt.ylabel(f"relative {method}-score")
        # plt.show()
        basepath = os.path.abspath('Project3/Results')
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
    return selector

def get_top_scored_features(scores, n=5):
    '''
    Returns a list of the indexes of the top n scores in 
    scores, sorted from largest to smallest.

    https://stackoverflow.com/questions/6910641
    '''
    max_scores_ind = np.argpartition(scores, -n)[-n:]
    sorted_indexes = np.flip(max_scores_ind[np.argsort(scores[max_scores_ind])])
    return sorted_indexes
    