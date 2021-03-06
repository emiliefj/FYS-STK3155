import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

import DecisionTree as dt


def read_data(file_location='../Data/Mushroom/UCI MLR/agaricus-lepiota.data.csv', 
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

