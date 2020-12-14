import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

import warnings

import DecisionTree as dt


class TreeEnsemble:
    def __init__(self, N_trees, n_samples=1,impurity_measure='gini', max_depth=5, min_samples_leaf=1, max_leaf_nodes=None, seed=77):
        from sklearn.tree import DecisionTreeClassifier
        np.random.seed(seed)
        self.seed = seed
        self.N_trees = N_trees
        self.n_samples = n_samples

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
        self.classes = np.unique(y)
        self.trees = []
        N = len(y) 
        n = self.n_samples
        if isinstance(n,float):
            n = len(y)*n

        for i in range(self.N_trees):
            # draw n data points 
            batch = np.random.randint(N,size=n)
            X_ = X[batch]
            y_ = y[batch]
            # shuffle data, with replacement
            # X_, y_ = resample(X, y) 
            # fit a decision tree to the data, save the resulting tree
            # tree = dt.DecisionTree(impurity_measure=self.measure, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.max_leaf_nodes,print_success=False)
            # tree.fit(X_,y_)

            # Using scikit-learn's decision tree for improved speed
            tree = DecisionTreeClassifier(criterion=self.measure, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.seed, max_leaf_nodes=self.max_leaf_nodes)
            tree.fit(X_train,y_train)
            self.trees.append(tree)

    def predict(self, X):
        '''
        Average prediction from each tree in ensemble.
        '''
        pred_sum = np.zeros((X.shape[0],len(self.classes)))
        for tree in self.trees:
            # current_prediction = tree.predict_probabilities(X)
            # pred_sum = pred_sum+current_prediction
            pred_sum = pred_sum+tree.predict_proba(X)
        return  self.classes[np.argmax(pred_sum,axis=1)]# pred_sum/len(self.trees)

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    

    data = load_breast_cancer()
    # data = load_digits()
    X,y,features = data['data'], data['target'],data['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

    # Compare result to using only one tree:
    bagging = Bagging(10,max_depth=5, min_samples_leaf=1, max_leaf_nodes=15)
    bagging.fit(X_train,y_train)
    y_pred = bagging.predict(X_train)

    print(f"\nTesting own bagging code tree code on breast cancer dataset:")
    print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    y_pred = bagging.predict(X_test)
    print("Test accuracy: ", dt.accuracy(y_pred,y_test))


    # tree = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50,max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)
    # tree.fit(X_train, y_train)
    # y_pred = tree.predict(X_train)

    # tree = dt.DecisionTree(max_depth=3, min_samples_leaf=1, max_leaf_nodes=15,print_success=False)
    # tree.fit(X_train,y_train)
    # # print(dt.get_tree_structure(tree,features))
    # y_pred = tree.predict(X_train)

    # print(f"\nComparing with a single decision tree using own code on breat cancer dataset:")
    # print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    # y_pred = tree.predict(X_test)
    # print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # from sklearn.tree import DecisionTreeClassifier
    # sk_tree = DecisionTreeClassifier(random_state=19)
    # sk_tree.fit(X_train,y_train)
    # y_pred = sk_tree.predict(X_train)
    # print()
    # print(f"\nTesting scikit-learn's DecisionTreeClassifier on  dataset:")
    # print("Train accuracy: ", dt.accuracy(y_pred,y_train))
    # y_pred = sk_tree.predict(X_test)
    # print("Test accuracy: ", dt.accuracy(y_pred,y_test))

    # from sklearn.tree import export_text
    # print()
    # print("The tree using scikitlearn's DecisionTreeClassifier:")
    # # print(export_text(sk_tree))





