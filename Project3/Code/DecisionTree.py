import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import warnings


class DecisionTree():
    '''
    A decision tree for classification.

    impurity_measure    - the chosen impurity measure used when 
                          building the tree
    max_depth           - the maximum depth of the created tree
    max_leaf_nodes      - the maximum number of leaf nodes in 
                          the created tree. Due to the way the 
                          tree is built the final numbar may 
                          surpass this
    seed                - integer for setting rng seed, used for
                          reproduceability
    print_success       - if True, the final depth and number of 
                          leaves is printed for the created tree.
    '''
    def __init__(self, impurity_measure='gini', max_depth=5, max_leaf_nodes=10,seed=71,print_success=True):
        np.random.seed(seed)
        self.measure = impurity_measure.lower()
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.print=print_success

        self.n_leaves = 0 # number of leaves as stopping criteria only sort of works
        self.depth = 0

        self.tree = None

    def fit(self, X, y):
        '''
        Builds a tree with the given data

        X   - the input data/features 
        y   - the target values
        '''
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        self.tree = Node()
        self.tree.depth = 0

        self.build_tree(X, y, self.tree)

        if self.print:
            print("\n* Created a decision tree with",self.n_leaves,"leaves, and a depth of", self.depth, "at the deepest.* ")


    def predict(self, X_test):
        '''
        Finds the predicted class of the data in X_test using
        the built decision tree

        X_test  - the test data to make prediciton for
        '''
        if not self.tree:
            raise Exception('Tree has not been fit to data. Run fit() first.')

        n_pred = X_test.shape[0]
        predictions = [] # For storing predictions

        for i in range(n_pred):
            predictions.append(self.classes[np.argmax(self.make_prediction(self.tree,X_test[i]))])

        return predictions


    def predict_probabilities(self, X_test):
        '''
        Finds the predicted probabilities for the data in X_test 
        using the built decision tree.

        X_test  - the test data to make prediciton for
        '''
        if not self.tree:
            raise Exception('Tree has not been fit to data. Run fit() first.')

        n_pred = X_test.shape[0]
        predictions = [] # For storing predictions

        for i in range(n_pred):
            predictions.append(self.make_prediction(self.tree,X_test[i]))

        return predictions

    def make_prediction(self,node,entry):
        '''
        Recursively moves down the tree and finds the list of 
        predictions/probabilities for each class for the given 
        entry.

        node    - current node
        entry   - value of the input value to make prediction for
        '''
        if(node.leaf): # reached leaf node, found prediction
            return node.prediction
        if isinstance(node.threshold, int) or isinstance(node.threshold,float):
            if(entry[node.feature]<node.threshold):
                return self.make_prediction(node.left, entry)
            else:
                return self.make_prediction(node.right, entry)
        else:
            if(entry[node.feature]==node.threshold):
                return self.make_prediction(node.left, entry)
            else:
                return self.make_prediction(node.right, entry)

    def make_leaf(self, node, y):
        '''
        make the current node a leaf node, update leaf count and
        store the class probabilities in this leaf node.

        node    - the node that is found to be a leaf
        y       - the outputs/targets of the training data
                  ending in this node
        '''
        node.leaf = True
        self.n_leaves = self.n_leaves+1
        self.set_prediction(node,y)

    def set_prediction(self,node,y):
        '''
        Sets the prediction for the node, meaning a list of
        predicted probabilitie for each possible class.

        node    - the (leaf) node to stor prediction for
        y       - the outputs/targets of the training data
                  ending in this node
        '''
        values, frequency = np.unique(y,  return_counts=True)
        probabilities = np.zeros(len(self.classes))
        indexes = [np.where(self.classes==values[i])[0][0] for i in range(len(values))]
        probabilities[indexes] = frequency/len(y)
        node.prediction = probabilities


    def build_tree(self, X, y, node):
        '''
        Recursively builds a decision tree from the input-node.
        Returns when the stopping criteria are met/it reaches a 
        leaf node.

        X    - the features/input data of the current node
        y    - the target/output data of the current node
        node - the current node to build a tree from
        '''

        if(node.depth>self.depth):
            self.depth = node.depth

        # Check stopping criteria #

        # reached max depth
        if node.depth >= self.max_depth:
            self.make_leaf(node,y)
            return
        
        # All entries belong to same class, making leaf
        if np.unique(y).shape[0] == 1:
            self.make_leaf(node,y)
            return

        # reached max number of leaf nodes
        if self.n_leaves>=(self.max_leaf_nodes-1):
            self.make_leaf(node,y)
            return

        # find best split for current node
        left, right, feature, threshold = self.find_split(X,y)

        if not left:
            # no split improving impurity found, making leaf
            self.make_leaf(node,y)
            return

        # create child nodes
        left_node, right_node = self.create_child_nodes(node, feature, threshold)

        # continue building tree
        self.build_tree(X[left], y[left], left_node)
        self.build_tree(X[right], y[right], right_node)
        
                


    def create_child_nodes(self, node, feature, threshold):
        left_node = Node()
        left_node.depth = node.depth + 1
        right_node = Node()
        right_node.depth = node.depth + 1
        node.left = left_node
        node.right = right_node
        node.feature = feature
        node.threshold = threshold

        return left_node, right_node

    def impurity_weight(self, z):
        if z==0:
            return 0
        else:
            return 1./z

    def find_split(self, X, y):
        '''
        Finds the split at the current node which reduces the impurity
        the most, if an improvement is found. Sets the associated 
        feature and threshold value for the node.
        '''

        start_impurity = self.calculate_impurity(y) 

        split_impurity = start_impurity

        split_threshold = None
        split_feature = None
        right = None
        left = None

        N = X.shape[0]
        N_features = X.shape[1]
        feature_loop = np.arange(N_features)
        np.random.shuffle(feature_loop)

        for i in feature_loop:
            unique_values, frequency = np.unique(X[:,i], return_counts=True)

            for val in unique_values:
                threshold = val
    
                # splitting the data using current value as threshold
                if isinstance(threshold, int) or isinstance(threshold,float):
                    left_index = np.where(X[:,i]<=val)
                    right_index = np.where(X[:,i]>val)
                else: # for string features equality is used
                    left_index = np.where(X[:,i]==val)
                    right_index = np.where(X[:,i]!=val)
                impurity_left = self.calculate_impurity(y[left_index])*len(left_index[0])/N
                impurity_right = self.calculate_impurity(y[right_index])*len(right_index[0])/N

                impurity = impurity_left + impurity_right
   
                if(impurity<split_impurity):
  
                    split_threshold = threshold
                    split_impurity = impurity
                    split_feature = i
                    left = left_index
                    right = right_index

        return left, right, split_feature, split_threshold

    def calculate_impurity(self, y):
        '''
        Calculates the impurity using the chosen impurity measure
        '''
        if(self.measure=='gini'):
            return self.gini_index(y)

        return

    def gini_index(self, y):
        '''
        Calculate the gini index/gini impurity

        gini = 1-sum(frequency^2)
        '''
        N = len(y)
        if(N==0):
            return 0

        classes, n_k = np.unique(y,  return_counts=True)
        
        return 1-1/(N**2)*np.sum(n_k**2)

    def entropy(self, y):
        pass


def get_tree_structure(tree, feature_names=None):
    """
    Recursively print the given tree. 
    Heavily inspired by scikit-learn's export_text()

    tree            - the DecisionTree object to print tree for
    feature_names   - The names of the features in the tree. If
                      not given the featured will be numbered
                      as feature_i for the ith feature.
    """
    if not tree.tree:
        raise Exception('Tree has not been fit to data. Run fit() first.')
        return 

    root = tree.tree
    
    if feature_names is None:
        features = ["feature_{}".format(i) for i in range(tree.n_features)]
    else:
        features = feature_names

    get_tree_structure.string = ""
    spacing = 3
    truncation_fmt = "{} {}\n"
    value_fmt = "{}{}{}\n" #value_fmt = "{}{}{}\n"

    def _add_leaf(probabilities, indent):
        classification = tree.classes[np.argmax(probabilities)]
        val = ''
        val += ' class: ' + str(classification) + ',    prediction: ' + str(["%.1f" % prob for prob in probabilities])
        get_tree_structure.string += value_fmt.format(indent, '', val)

    
    def recursively_print_tree(node, depth):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing
        probabilities = node.prediction
    
        info_fmt = ""
        info_fmt_left = info_fmt
        info_fmt_right = info_fmt
        if node.leaf:
            _add_leaf(probabilities, indent)
        else:
            name = features[node.feature]
            threshold = node.threshold
            if isinstance(threshold, int) or isinstance(threshold,float):
                threshold = "{1:.{0}f}".format(2, threshold)
                right_child_fmt = "{} {} <= {}\n"
                left_child_fmt = "{} {} >  {}\n"
            else:
                threshold = "{:d}".format(threshold)
                right_child_fmt = "{} {} == {}\n"
                left_child_fmt = "{} {} !=  {}\n"
            get_tree_structure.string += right_child_fmt.format(indent,
                                                         name,
                                                         threshold)
            get_tree_structure.string += info_fmt_left
            recursively_print_tree(node.left, depth+1)
            get_tree_structure.string += left_child_fmt.format(indent,
                                                        name,
                                                        threshold)
            get_tree_structure.string += info_fmt_right
            recursively_print_tree(node.right, depth+1)

    recursively_print_tree(root, 1)
    return get_tree_structure.string


class Node():
    '''

    feature     - The feature the node makes its decision based on
    threshold   - The threshold separating the left and right branch
                  of the node
    '''
    def __init__(self):
        
        self.feature = None
        self.threshold = None
        self.prediction = None
        self.right = None
        self.left = None

        self.depth = None
        self.leaf = None
        

def accuracy(pred, actual):
    ''' 
    A function for measuring the accuracy of classification
    Returns an accuracy score as the fraction of predictions that are 
    correct.

    Accuracy = sum(correct predictions)/number of prediction

    pred    - the prediction made by the model
    actual  - the actual value in the data 
    '''
    n = len(pred)
    correctly_predicted = 0
    for i in range(n):
        if(pred[i]==actual[i]):
            correctly_predicted += 1

    return correctly_predicted/n


#####################################################################
############# Tests and runs for specific datasets ##################
#####################################################################



def test_breast_cancer(plot=False, print_tree=False, random=13):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    compare_trees_dataframe(data,random=random,name="breast cancer",plot=plot, print_tree=print_tree, feature_names=data['feature_names'])

def test_iris(plot=False, print_tree=False, random=13):
    from sklearn.datasets import load_iris

    data = load_iris()
    # print(data['feature_names'])
    compare_trees_dataframe(data,random=random,name="iris", plot=plot, print_tree=print_tree, feature_names=data['feature_names'])

def compare_trees_dataframe(data, random=13, name="iris", plot=False, print_tree=False, feature_names=None):
    from sklearn.model_selection import train_test_split

    X,y,features = data['data'], data['target'],data['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random)
    compare_trees(X_train, y_train, X_test, y_test, random=random, name=name, plot=plot,print_tree=print_tree, feature_names=feature_names)


def compare_trees(X_train,y_train,X_test,y_test, max_depth=5, max_leaf_nodes=10, random=13, name="iris", plot=False, print_tree=False, feature_names=None):
    from sklearn.model_selection import train_test_split

    build_and_test_tree(X_train,y_train,X_test,y_test,max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name=name, print_tree=print_tree, feature_names=feature_names)
    test_scikit_tree(X_train,y_train,X_test,y_test,max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random=random, name=name, plot=plot, print_tree=print_tree, feature_names=feature_names)


def test_scikit_tree(X_train, y_train, X_test, y_test, max_depth=5, max_leaf_nodes=10, random=13, name="iris", plot=False, print_tree=False, feature_names=None):
    from sklearn.tree import DecisionTreeClassifier
    sk_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random)
    sk_tree.fit(X_train,y_train)
    y_pred = sk_tree.predict(X_train)
    print()
    print(f"\nTesting scikit-learn's DecisionTreeClassifier on {name} dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}:")
    print("Train accuracy: ", accuracy(y_pred,y_train))
    y_pred = sk_tree.predict(X_test)
    print("Test accuracy: ", accuracy(y_pred,y_test))

    if plot:
        from sklearn.tree import plot_tree
        plot_tree(sk_tree)
        plt.show()

    if print_tree:
        from sklearn.tree import export_text
        print()
        print("The tree using scikitlearn's DecisionTreeClassifier:")
        print(export_text(sk_tree, feature_names=list(feature_names)))

        
def build_and_test_tree(X_train, y_train, X_test, y_test, max_depth=5, max_leaf_nodes=10, random=13, name="iris", print_tree=False, feature_names=None, print_success=True):

    tree = DecisionTree(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,seed=random,print_success=print_success)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_train)

    print(f"\nTesting own decision tree code on {name} dataset with max_depth = {max_depth} and max_leaf_nodes = {max_leaf_nodes}:")
    print("Train accuracy: ", accuracy(y_pred,y_train))
    y_pred = tree.predict(X_test)
    print("Test accuracy: ", accuracy(y_pred,y_test))

    if print_tree:
        print()
        print("The tree using my own code:")
        print(get_tree_structure(tree,feature_names=feature_names))
    return tree


if __name__ == '__main__':

    

    test_iris(plot=False,print_tree=True,random=13)
    print()
    test_breast_cancer(print_tree=True)
    





