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
    
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_train_val_test(X[1:],y[1:],0.6,0.2,0.2,seed=99)


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

    # 3. Testing model #

    # First fit decision tree to all features, as well as feature as found in feature selection, to compare.
    # Comparison tests are done with validation data
 
    # Tree using own code fit to all data
    # full_tree=dt.build_and_test_tree(X_train,y_train,X_val,y_val, max_depth=5, max_leaf_nodes=10, random=15, name="fetal health", print_tree=True, feature_names=features)
    
    # Tree fit to all data using own code and scikit-learn DecisionTreeClassifier to compare
    print("\nComparing own code and scikit-learn on full fetal health dataset:")
    dt.compare_trees(X_train,y_train,X_test,y_test, max_depth=5, max_leaf_nodes=10, random=15, name="fetal health", plot=False, print_tree=False, feature_names=features)

    # Using features selected using variance threshold
    print("\nDecision tree with features selected using variance threshold:")
    variance_threshold_tree=dt.build_and_test_tree(X_train[:,chosen_indexes],y_train,X_val[:,chosen_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="fetal health", print_tree=False, feature_names=features[chosen_indexes])

    # Using features selected using univariate feature selection with f_classif
    print("\nDecision tree with features selected using f_classif:")
    f_tree=dt.build_and_test_tree(X_train[:,f_top_indexes],y_train,X_val[:,f_top_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="fetal health", print_tree=False, feature_names=features[f_top_indexes])

    # Using features selected using univariate feature selection with mutual information
    print("\nDecision tree with features selected using mutual information:")
    mutual_info_tree=dt.build_and_test_tree(X_train[:,mutual_top_indexes],y_train,X_val[:,mutual_top_indexes],y_val, max_depth=5, max_leaf_nodes=10, random=13, name="fetal health", print_tree=False, feature_names=features[mutual_top_indexes])

    # # Compare to logistic regression

    # # transform target to take values 0, 1, 2 instead of 1.0, 2.0, 3.0
    # y_train = (y_train-1).astype(int)
    # y_val = (y_val-1).astype(int)

    # best_lr = lr.find_learning_rate(X_train,y_train,X_val,y_val,k=3,plot=True)
    # logreg = lr.LogisticRegression(lmda=0, decay=False, seed=7)
    # n_epochs = 100
    # batchsize = 50
    # logreg.sgd(X_train, y_train, n_epochs=n_epochs, batchsize=batchsize, learning_rate=best_lr, n_classes=3, print_epochs=False)
    # pred_train = logreg.predict(X_train)
    # pred_val = logreg.predict(X_val)

    # print(f'Accuracy on fetal health dataset using logistic regression with a learning rate of {best_lr}, {n_epochs} epochs and a batchsize of {batchsize} is: \nTraining: {lr.accuracy(pred_train, y_train)}\nValidation: {lr.accuracy(pred_val, y_val)}\n')