import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
import sys
sys.path.append('../Code')
sys.path.append('../Data')
import DecisionTree as dt


def read_data(file_location='../Data/UCI MLR/agaricus-lepiota.data.csv', 
              names=['class','cap-shape','cap-surface','cap-color',
                 'bruises','odor','gill-attachment','gill-spacing',
                 'gill-size','gill-color','stalk-shape','stalk-root',
                 'stalk-surface-above-ring','stalk-surface-below-ring',
                 'stalk-color-above-ring','stalk-color-below-ring',
                 'veil-type','veil-color','ring-number','ring-type',
                 'spore-print-color','population','habitat']):

    return pd.read_csv(file_location, 
                 names=names)

def make_df_numerical(df):
    '''
    Map categorical values in dataframe to numerical values for 
    easier processing.

    df  - the dataframe with categorical values
    '''
    from sklearn.preprocessing import LabelEncoder
    les = {col: LabelEncoder() for col in df.columns}
    for col in les:
        df[col] = les[col].fit_transform(df[col])
    return df


def split_train_val_test(X, y, train=0.6, val=0.2, test=0.2, seed=67):
    '''
    Split the input dataset into training, validation and test subsets.
    Returns X_train, X_validation, X_test, y_train, y_validetion, y_test
    The split is stratified, to preserve the proportion of targets in
    each subset

    X       - the input features/design matrix
    y       - the output/target
    train,val,test - the fractions for each part of the set
    seed    - seed for reproduceability   
    '''
    total = train+val+test
    if total!=1:
        warnings.warn(f'\nFractions do not add up to 1.0: {train}+{val}+{test}={total}. Using default values train=0.6, val=0.2, test=0.2\n')
        train, val, test = 0.6, 0.2, 0.2
    val_and_test = 1-train
    test_frac = test/val_and_test
    val_frac = val/val_and_test

    X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(X, y, train_size=train, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test_and_val, y_test_and_val, test_size=test_frac, train_size=val_frac, random_state=seed*3, stratify=y_test_and_val)

    return X_train, X_val, X_test, y_train, y_val, y_test



def variance_threshold(X_train, cutoff=0.8):
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=(cutoff*(1-cutoff)))
    selector.fit_transform(X_train)

    return selector.get_support()


def feature_selection(X_train,y_train,plot=False):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    selector = SelectKBest(chi2, k='all')
    selector.fit(X_train, y_train)
    scores = selector.scores_
    # log_scores = scores#-np.log10(scores)
    if plot:
        scores /= scores.max()
        fig, ax = plt.subplots(figsize =(12, 8)) 
        plt.bar(np.arange(X.shape[-1]), scores, width=.2)
        plt.title("The relative chi2 scores for each feature")
        plt.xlabel("feature index")
        plt.ylabel("relative chi2-score")
        plt.show()
    return selector



def print_array(x,opt='s'):
    option_str = '{:'+opt+'}'
    return [option_str.format(i) for i in x]

if __name__ == '__main__':

    # 1. Pre-process data #
    #print("We begin by reading in data, and splitting into train, validation, and test subsets.")

    df = read_data()

    # odor = df['odor']
    # indexes = np.array(np.where((odor!='n') & (odor!= 'a') & (odor!='l')))
    # y = df['class']
    # unique, frequency = np.unique(y, return_counts=True) 
    # print(unique,"  ", frequency)

    # indexes = indexes.reshape((-1,))
    # result = y[indexes]

    # unique, frequency = np.unique(result,return_counts=True) 
    # print(unique,"  ", frequency)


    # 'veil-type' has zero variance
    df.drop(columns=['veil-type'], axis=1, inplace=True)
    features = np.array(df.columns[1:])
    df = make_df_numerical(df)


    y = df['class']
    X = df.drop(['class'], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X,y,0.6,0.2,0.2)


    # 2. Feature selection #

    # Variance threshold
    cutoff = 0.8
    chosen_indexes = variance_threshold(X_train, cutoff)
    
    chosen_features = features[chosen_indexes] # +1 as 'class' is removed at 0
    excluded_features = features[np.where(chosen_indexes==False)]
    print(f"The {len(chosen_features)} features selected when using a variance threshold of {cutoff} as cutoff are: {print_array(chosen_features)} leaving out the features: {print_array(excluded_features)}")

    # Univariate feature selection
    # Issue: I can get 98.52% accuracy using just 'odor'. yet 
    # odor is not selected as an important feature
    selector = feature_selection(X_train,y_train,plot=False)
    # print(features[np.argmax(selector.scores_)])
    # print()
    # index_best_ordered = np.array([8,17,7,3,10,20,6])
    # print(features[index_best_ordered])
    # print(selector.scores_[index_best_ordered])

    # print(np.where(X_train[5]!='n'))
    # print(X_train)



     

