# imports:
import numpy as np 
import pandas as pd

# Own code
import DecisionTree as dt
import preprocess_data as pp
import feature_selection as fs

def print_array(x,opt='s'):
    option_str = '{:'+opt+'}'
    return [option_str.format(i) for i in x]



if __name__ == '__main__':

    # 1. Pre-process data #
   
    df = pp.read_data()

    # 'veil-type' has zero variance
    df.drop(columns=['veil-type'], axis=1, inplace=True)
    features = np.array(df.columns[1:])
    # df = pp.make_df_numerical(df)


    y = np.array(df['class'])
    X = np.array(df.drop(['class'], axis=1))

    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_train_val_test(X[1:],y[1:],0.6,0.2,0.2)


    # 2. Feature selection #

    # # Variance threshold
    # cutoff = 0.8
    # chosen_indexes = fs.variance_threshold(X_train, cutoff)
    
    # chosen_features = features[chosen_indexes]
    # excluded_features = features[np.where(chosen_indexes==False)]
    # print(f"The {len(chosen_features)} features selected when using a variance threshold of {cutoff} as cutoff are: {print_array(chosen_features)} leaving out the features: {print_array(excluded_features)}\n")

    # # Univariate feature selection
    # chi2_selector = fs.feature_selection(X_train,y_train,method='chi2',plot=False)
    # chi2_top_indexes = fs.get_top_scored_features(chi2_selector.scores_, n=7)
    # chi2_selected_features = features[chi2_top_indexes]
    # print(f"The top {len(chi2_selected_features)} features according to their chi-squared score are: {print_array(chi2_selected_features)}.\n")

    # mutual_selector = fs.feature_selection(X_train,y_train,method='mutual_info',plot=False)
    # mutual_top_indexes = fs.get_top_scored_features(mutual_selector.scores_, n=13)
    # mutual_selected_features = features[mutual_top_indexes]
    # print(f"The top {len(mutual_selected_features)} features according to their mutual information score are: {print_array(mutual_selected_features)}.\n")

    # 3. Testing model #

    # First fit decision tree to all features, as well as feature as found in feature selection, to compare.
    # Comparison tests are done with validation data

    # Tree fit to all data
    full_tree=dt.build_and_test_tree(X_train,y_train,X_val,y_val, max_depth=10, max_leaf_nodes=10, random=13, name="mushroom", print_tree=True, feature_names=features)
    prediction = full_tree.predict(X_val)




