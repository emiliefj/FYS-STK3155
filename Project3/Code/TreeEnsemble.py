import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

import warnings

import DecisionTree as dt


class TreeEnsemble:
    def __init__(self, N_trees, n_samples=0.5, n_features=None, impurity_measure='gini', max_depth=5, min_samples_leaf=1, max_leaf_nodes=None,  seed=77):
        from sklearn.tree import DecisionTreeClassifier
        np.random.seed(seed)
        self.seed = seed
        self.N_trees = N_trees
        self.n_samples = n_samples
        self.n_features = n_features

        # For each tree:
        self.measure = impurity_measure.lower()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class Bagging(TreeEnsemble):

    def fit(self, X, y):
        '''
        Fits each of the N_trees to a subset of the input data X and
        y. The data to fit to is a subset of size n_samples drawn 
        randomly from X and y for each tree. The trees are stored 
        in a list for future predictions.
        I am using scikit-learn's DecisionTreeClassifier in place of 
        my own DecisionTree as the latter is not optimized for speed.

        X   - the features of the training data
        y   - the target of the training data
        '''
        self.classes = np.unique(y)
        self.trees = []
        N = len(y) 
        n = self.n_samples
        if isinstance(n,float):
            n = int(len(y)*n)

        for i in range(self.N_trees):
            # draw n data points 
            batch = np.random.randint(N,size=n)
            X_ = X[batch]
            y_ = y[batch]

            # Using scikit-learn's decision tree for improved speed
            tree = DecisionTreeClassifier(criterion=self.measure, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.seed, max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(X_,y_)
            self.trees.append(tree)

    def predict(self, X):
        '''
        Average prediction from each tree in ensemble.

        X   - The data to make prediction for
        '''
        pred_sum = np.zeros((X.shape[0],len(self.classes)))
        for tree in self.trees:
            pred_sum = pred_sum+tree.predict_proba(X)
        return  self.classes[np.argmax(pred_sum,axis=1)]

class RandomForest(Bagging):

    def fit(self, X, y):
        '''
        Fits each of the N_trees to a subset of the input data X and
        y. The data to fit to is a subset of size n_samples drawn 
        randomly from X and y for each tree. At each fit only a 
        selection, n_features of the features are used in the fitting.
        The trees are stored in a list for future predictions.
        I am using scikit-learn's DecisionTreeClassifier in place of 
        my own DecisionTree as the latter is not optimized for speed.

        X   - the features of the training data
        y   - the target of the training data
        '''
        self.classes = np.unique(y)
        self.trees = []
        self.used_features = []
        N, n_feat = X.shape 
        n = self.n_samples
        if isinstance(n,float):
            n = int(N*n)
        f = self.n_features
        if f is None:
            f = int(sqrt(n_feat))
        elif isinstance(f,float):
            f = int(n_feat*f)

        for i in range(self.N_trees):
            # draw n data points 
            batch = np.random.randint(N,size=n)
            features = np.random.randint(n_feat,size=f)
            X_ = X[batch]
            X_ = X_[:,features]
            y_ = y[batch]

            # Using scikit-learn's decision tree for improved speed
            tree = DecisionTreeClassifier(criterion=self.measure, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.seed, max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(X_,y_)
            self.trees.append(tree)
            self.used_features.append(features)

    def predict(self, X):
        '''
        Average prediction from each tree in ensemble.

        X   - The data to make prediction for
        '''
        pred_sum = np.zeros((X.shape[0],len(self.classes)))
        for i in range(self.N_trees):
            X_ = X[:,self.used_features[i]]
            pred_sum = pred_sum+self.trees[i].predict_proba(X_)
        return  self.classes[np.argmax(pred_sum,axis=1)]


if __name__ == '__main__':

    ### Testing the Ensembles on scikit-learn's breast cancer dataset ###

    # Imports
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Load and split data
    data = load_breast_cancer()
    X,y,features = data['data'], data['target'],data['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

    n_trees = 100
    n_samples = 1.0
    n_features = 5
    max_depth = 3
    min_samples_leaf=1
    max_leaf_nodes=10
    random = 99

    # Own bagging code
    bagging = Bagging(n_trees,n_samples=n_samples,max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    bagging.fit(X_train,y_train)
    y_pred = bagging.predict(X_train)
    print(f"\nTesting bagging code on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = bagging.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # Own random forest code
    rf = RandomForest(n_trees,n_samples=n_samples,n_features=n_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,seed=random)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_train)
    print(f"\nTesting random forest code on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = rf.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # One tree
    sk_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,random_state=random)
    sk_tree.fit(X_train,y_train)
    y_pred = sk_tree.predict(X_train)
    print(f"\nTesting scikit-learn's DecisionTreeClassifier on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = sk_tree.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # scikit-learn's Bagging module
    tree = BaggingClassifier(sk_tree, n_estimators=n_trees,max_samples=n_samples, bootstrap=True, n_jobs=-1, random_state=random)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    print(f"\nTesting scikit-learn's BaggingClassifier on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = tree.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # scikit-learn's RandomForestClassifier module
    tree = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, max_samples=None, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, max_features='auto', bootstrap=True, n_jobs=-1, random_state=random)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    print(f"\nTesting scikit-learn's RandomForestClassifier on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = tree.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))






