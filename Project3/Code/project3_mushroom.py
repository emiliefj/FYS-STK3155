# imports:
import numpy as np 
import pandas as pd

# Own code
import DecisionTree as dt
import LogisticRegression as lr
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
    df = pp.make_df_numerical(df)


    y = np.array(df['class'])
    X = np.array(df.drop(['class'], axis=1))

    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_train_val_test(X[1:],y[1:],0.6,0.2,0.2)
    # print(X_train)


    # 2. Feature selection #

    # Variance threshold
    cutoff = 0.8
    chosen_indexes = fs.variance_threshold(X_train, cutoff)
    
    chosen_features = features[chosen_indexes]
    excluded_features = features[np.where(chosen_indexes==False)]
    print(f"The {len(chosen_features)} features selected when using a variance threshold of {cutoff} as cutoff are: {print_array(chosen_features)} leaving out the features: {print_array(excluded_features)}\n")

    # Univariate feature selection
    chi2_selector = fs.feature_selection(X_train,y_train,method='chi2',plot=False)
    chi2_top_indexes = fs.get_top_scored_features(chi2_selector.scores_, n=7)
    chi2_selected_features = features[chi2_top_indexes]
    print(f"The top {len(chi2_selected_features)} features according to their chi-squared score are: {print_array(chi2_selected_features)}.\n")

    mutual_selector = fs.feature_selection(X_train,y_train,method='mutual_info',plot=False)
    mutual_top_indexes = fs.get_top_scored_features(mutual_selector.scores_, n=13)
    mutual_selected_features = features[mutual_top_indexes]
    print(f"The top {len(mutual_selected_features)} features according to their mutual information score are: {print_array(mutual_selected_features)}.\n")

    # 3. Testing model #

    # First fit decision tree to all features, as well as feature as found in feature selection, to compare.
    # Comparison tests are done with validation data
 
    # Tree using own code fit to all data
    # full_tree=dt.build_and_test_tree(X_train,y_train,X_val,y_val, max_depth=10, max_leaf_nodes=10, random=13, name="mushroom", print_tree=False, feature_names=features)
    
    # Tree fit to all data using own code and scikit-learn DecisionTreeClassifier to compare
    dt.compare_trees(X_train,y_train,X_val,y_val, max_depth=5, max_leaf_nodes=10, random=13, name="mushroom", plot=False, print_tree=False, feature_names=features)

    # Using features selected using variance threshold
    variance_threshold_tree=dt.build_and_test_tree(X_train[:,chosen_indexes],y_train,X_val[:,chosen_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="mushroom", print_tree=False, feature_names=features[chosen_indexes])

    # Using features selected using univariate feature selection with chi2
    chi2_tree=dt.build_and_test_tree(X_train[:,chi2_top_indexes],y_train,X_val[:,chi2_top_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="mushroom", print_tree=False, feature_names=features[chi2_top_indexes])

    # Using features selected using univariate feature selection with mutual information
    mutual_info_tree=dt.build_and_test_tree(X_train[:,mutual_top_indexes],y_train,X_val[:,mutual_top_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="mushroom", print_tree=False, feature_names=features[mutual_top_indexes])

    # Compare to logistic regression
    best_lr = lr.find_learning_rate(X_train,y_train,X_val,y_val,k=2,plot=False)
    logreg = lr.LogisticRegression(lmda=0, decay=False, seed=7)
    n_epochs = 100
    batchsize = 50
    logreg.sgd(X_train, y_train, n_epochs=n_epochs, batchsize=batchsize, learning_rate=best_lr, n_classes=2, print_epochs=False)
    pred_train = logreg.predict(X_train)
    pred_val = logreg.predict(X_val)

    print(f'Accuracy on mushroom dataset using logistic regression with a learning rate of {best_lr}, {n_epochs} epochs and a batchsize of {batchsize} is: \nTraining: {lr.accuracy(pred_train, y_train)}\nValidation: {lr.accuracy(pred_val, y_val)}\n')




