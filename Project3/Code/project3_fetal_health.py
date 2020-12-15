# imports:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Own code
import DecisionTree as dt
import TreeEnsemble as te
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

    # transform target to take values 0, 1, 2 instead of 1.0, 2.0, 3.0
    y_train = (y_train-1).astype(int)
    y_val = (y_val-1).astype(int)
    y_test = (y_test-1).astype(int)


    # Feature selection # 

    # # Variance threshold
    # cutoff = 0.8
    # chosen_indexes = fs.variance_threshold(X_train, cutoff)
    
    # chosen_features = features[chosen_indexes]
    # excluded_features = features[np.where(chosen_indexes==False)]
    # print(f"The {len(chosen_features)} features selected when using a variance threshold of {cutoff} as cutoff are: {print_array(chosen_features)} leaving out the features: {print_array(excluded_features)}\n")

    # # Univariate feature selection

    # f_selector = fs.feature_selection(X_train,y_train,method='f_classif',plot=False)
    # f_top_indexes = fs.get_top_scored_features(f_selector.scores_, n=15)
    # f_selected_features = features[f_top_indexes]
    # print(f"The top {len(f_selected_features)} features according to their ANOVA F-value score are: {print_array(f_selected_features)}.\n")

    # mutual_selector = fs.feature_selection(X_train,y_train,method='mutual_info',plot=False)
    # mutual_top_indexes = fs.get_top_scored_features(mutual_selector.scores_, n=17)
    # mutual_selected_features = features[mutual_top_indexes]
    # print(f"The top {len(mutual_selected_features)} features according to their mutual information score are: {print_array(mutual_selected_features)}.\n")

    # 3. Testing model #

    max_depth = 10
    max_leaf_nodes = 100
    random = 13

    # First fit decision tree to all features, as well as feature as found in feature selection, to compare.
    # Comparison tests are done with validation data
 
    # Tree using own code fit to all data
    full_tree=dt.build_and_test_tree(X_train,y_train,X_val,y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="fetal health", print_tree=False, feature_names=features)
    
    # Tree fit to all data using own code and scikit-learn DecisionTreeClassifier to compare
    # print("\n_Comparing own code and scikit-learn on full fetal health dataset:_")
    # dt.compare_trees(X_train,y_train,X_test,y_test, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="fetal health", plot=False, print_tree=False, feature_names=features)

    # # Using features selected using variance threshold
    # print("\n\n_Decision tree with features selected using variance threshold:_")
    # variance_threshold_tree=dt.build_and_test_tree(X_train[:,chosen_indexes],y_train,X_val[:,chosen_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="fetal health", print_tree=False, feature_names=features[chosen_indexes])

    # # Using features selected using univariate feature selection with f_classif
    # print("\n\n_Decision tree with features selected using f_classif:_")
    # f_tree=dt.build_and_test_tree(X_train[:,f_top_indexes],y_train,X_val[:,f_top_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="fetal health", print_tree=False, feature_names=features[f_top_indexes])

    # # Using features selected using univariate feature selection with mutual information
    # print("\n\n_Decision tree with features selected using mutual information:_")
    # mutual_info_tree=dt.build_and_test_tree(X_train[:,mutual_top_indexes],y_train,X_val[:,mutual_top_indexes],y_val, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name="fetal health", print_tree=False, feature_names=features[mutual_top_indexes])

    # # Compare to logistic regression
    # best_lr = lr.find_learning_rate(X_train,y_train,X_val,y_val,k=3,plot=False)
    # logreg = lr.LogisticRegression(lmda=0, decay=False, seed=7)
    # n_epochs = 100
    # batchsize = 50
    # logreg.sgd(X_train, y_train, n_epochs=n_epochs, batchsize=batchsize, learning_rate=best_lr, n_classes=3, print_epochs=False)
    # pred_train = logreg.predict(X_train)
    # pred_val = logreg.predict(X_val)

    # print(f'\nAccuracy on fetal health dataset using logistic regression with a learning rate of {best_lr}, {n_epochs} epochs and a batchsize of {batchsize} is: \nTraining: {lr.accuracy(pred_train, y_train)}\nValidation: {lr.accuracy(pred_val, y_val)}\n')

    # Ensemble Methods #

    n_trees = 100
    n_samples = 0.5
    n_features = 5
    min_samples_leaf=1

    # Own bagging code
    bagging = te.Bagging(n_trees,n_samples=n_samples,max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    bagging.fit(X_train,y_train)
    y_pred = bagging.predict(X_train)
    print(f"\nTesting bagging code on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}, using a fraction {n_samples} of the total training data at each fitting:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = bagging.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))

    # Own random forest code
    rf = te.RandomForest(n_trees,n_samples=n_samples,n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_train)
    print(f"\nTesting random forest code on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}, using a fraction {n_samples} of the total training data and {n_features} features at each fitting:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = rf.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))

    # Own Adaptive Boosting code
    ab = te.AdaptiveBoosting(n_trees,n_samples=n_samples, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    ab.fit(X_train,y_train)
    y_pred = ab.predict(X_train)
    print(f"\nTesting adaptive boosting code on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = ab.predict(X_val)
    print("Test accuracy: ", dt.accuracy(y_pred,y_val))


    # 4. Final test of model performance on test data #
    
    import scikitplot as skplt

    # # Single decision tree
    # y_pred = full_tree.predict(X_test)
    # y_probas = full_tree.predict_probabilities(X_test)
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    # plt.show()  
    # skplt.metrics.plot_roc(y_test, y_probas)
    # plt.show()

    # # Adaptive boosting tree
    # y_pred = ab.predict(X_test)
    # y_probas = ab.predict_probabilities(X_test)
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    # plt.show()  
    # skplt.metrics.plot_roc(y_test, y_probas)
    # plt.show()

    # # Bagging
    # y_pred = bagging.predict(X_test)
    # y_probas = bagging.predict_probabilities(X_test)
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    # plt.show()  
    # skplt.metrics.plot_roc(y_test, y_probas)
    # plt.show()

    # Own Decision Tree code
    y_pred = full_tree.predict(X_test)
    print(f"\nSingle DecisionTree on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes} on test set:")
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))


    # Adaptive boosting code
    y_pred = ab.predict(X_test)
    print(f"\nAdaptive Boosting code on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes} on test set:")
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # Own Bagging code
    y_pred = bagging.predict(X_test)
    print(f"\nBagging code on fetal health dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes} on test set:")
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))
