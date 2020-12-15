# imports:
import numpy as np 
import pandas as pd

# Own code
import DecisionTree as dt
import TreeEnsemble as te
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
    mutual_top_indexes = fs.get_top_scored_features(mutual_selector.scores_, n=14)
    mutual_selected_features = features[mutual_top_indexes]
    print(f"The top {len(mutual_selected_features)} features according to their mutual information score are: {print_array(mutual_selected_features)}.\n")

    # 3. Testing model #

    max_depth = 1
    max_leaf_nodes = 2
    random = 13

    # First fit decision tree to all features, as well as feature as found in feature selection, to compare.
    # Comparison tests are done with validation data
 
    # Tree using own code fit to all data
    # full_tree=dt.build_and_test_tree(X_train,y_train,X_val,y_val, max_depth=10, max_leaf_nodes=10, random=13, name="mushroom", print_tree=False, feature_names=features)
    
    # Tree fit to all data using own code and scikit-learn DecisionTreeClassifier to compare
    print("\nAll features included in fit:")
    dt.compare_trees(X_train,y_train,X_val,y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="mushroom", plot=False, print_tree=False, feature_names=features)

    # Using features selected using variance threshold
    print("\n\nUsing features selected using variance threshold:")
    variance_threshold_tree=dt.build_and_test_tree(X_train[:,chosen_indexes],y_train,X_val[:,chosen_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="mushroom", print_tree=False, feature_names=features[chosen_indexes])

    # Using features selected using univariate feature selection with chi2
    print("\n\nUsing features selected using univariate feature selection with chi2:")
    chi2_tree=dt.build_and_test_tree(X_train[:,chi2_top_indexes],y_train,X_val[:,chi2_top_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="mushroom", print_tree=False, feature_names=features[chi2_top_indexes])

    # Using features selected using univariate feature selection with mutual information
    print("\n\nUsing features selected using univariate feature selection with mutual information:")
    mutual_info_tree=dt.build_and_test_tree(X_train[:,mutual_top_indexes],y_train,X_val[:,mutual_top_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="mushroom", print_tree=False, feature_names=features[mutual_top_indexes])

    # Compare to logistic regression
    best_lr = lr.find_learning_rate(X_train,y_train,X_val,y_val,k=2,plot=False)
    logreg = lr.LogisticRegression(lmda=0, decay=False, seed=7)
    n_epochs = 100
    batchsize = 50
    logreg.sgd(X_train, y_train, n_epochs=n_epochs, batchsize=batchsize, learning_rate=best_lr, n_classes=2, print_epochs=False)
    pred_train = logreg.predict(X_train)
    pred_val = logreg.predict(X_val)

    print(f'Accuracy on mushroom dataset using logistic regression with a learning rate of {best_lr}, {n_epochs} epochs and a batchsize of {batchsize} is: \nTraining: {lr.accuracy(pred_train, y_train)}\nValidation: {lr.accuracy(pred_val, y_val)}\n')


    # Ensemble Methods #

    n_trees = 100
    n_samples = 0.5
    n_features = 5
    min_samples_leaf=1

    # Own bagging code
    bagging = te.Bagging(n_trees,n_samples=n_samples,max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    bagging.fit(X_train,y_train)
    y_pred = bagging.predict(X_train)
    print(f"\nTesting bagging code on mushroom dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}, using a fraction {n_samples} of the total training data at each fitting:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = bagging.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))

    # Own random forest code
    rf = te.RandomForest(n_trees,n_samples=n_samples,n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_train)
    print(f"\nTesting random forest code on mushroom dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}, using a fraction {n_samples} of the total training data and {n_features} features at each fitting:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = rf.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))

    # Own Adaptive Boosting code
    ab = te.AdaptiveBoosting(n_trees,n_samples=n_samples, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    ab.fit(X_train,y_train)
    y_pred = ab.predict(X_train)
    print(f"\nTesting adaptive boosting code on mushroom dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = ab.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))

