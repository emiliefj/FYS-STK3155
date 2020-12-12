# imports:
import numpy as np 
import pandas as pd

# Own code
import DecisionTree as dt
import LogisticRegression as lr
import preprocess_data as pp
import feature_selection as fs

def print_array(x,opt='s'): # move to tools?
    option_str = '{:'+opt+'}'
    return [option_str.format(i) for i in x]

if __name__ == '__main__':
    
    # Pre-process data #

    df = pp.read_data('../Data/Fetal Health/fetal_health.csv', names=None)

    # drop histogram_mean and histogram_median in favor of histogram_mode
    df.drop(columns=['histogram_mean'], axis=1, inplace=True)
    df.drop(columns=['histogram_median'], axis=1, inplace=True)

    features = np.array(df.columns[:-1])
    y = np.array(df['fetal_health'])
    X = np.array(df.drop(['fetal_health'], axis=1))
    
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_train_val_test(X[1:],y[1:],0.6,0.2,0.2)


    # Feature selection # 

    # Variance threshold
    cutoff = 0.8
    chosen_indexes = fs.variance_threshold(X_train, cutoff)
    
    chosen_features = features[chosen_indexes]
    excluded_features = features[np.where(chosen_indexes==False)]
    print(f"The {len(chosen_features)} features selected when using a variance threshold of {cutoff} as cutoff are: {print_array(chosen_features)} leaving out the features: {print_array(excluded_features)}\n")

    # Univariate feature selection

    f_selector = fs.feature_selection(X_train,y_train,method='f_classif',plot=False)
    f_top_indexes = fs.get_top_scored_features(f_selector.scores_, n=15)
    f_selected_features = features[f_top_indexes]
    print(f"The top {len(f_selected_features)} features according to their ANOVA F-value score are: {print_array(f_selected_features)}.\n")

    mutual_selector = fs.feature_selection(X_train,y_train,method='mutual_info',plot=False)
    mutual_top_indexes = fs.get_top_scored_features(mutual_selector.scores_, n=17)
    mutual_selected_features = features[mutual_top_indexes]
    print(f"The top {len(mutual_selected_features)} features according to their mutual information score are: {print_array(mutual_selected_features)}.\n")