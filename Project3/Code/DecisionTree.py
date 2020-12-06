import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class DecisionTree():

    def __init__():
        pass


class Node():
    '''

    feature     - The feature the node makes its deciosion based on
    threshold   - The threshold separating the left and right branch of the node

    '''
    def __init__(self, feature, threshold):
        

        self.feature = feature
        self.threshold = threshold
        self.right = None
        self.left = None

        self.depth = None
        self.leaf = None
        pass


    def split_node(self,split_feature, split_threshold, input_data, target):
        pass


class DecisionTree():
    '''
    A decision tree for classification.


    '''
    def __init__(self, impurity_measure='gini', max_depth=5, min_samples_leaf=1, max_leaf_nodes=20):
        
        self.measure = impurity_measure.lower()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

        self.tree = None

    def fit(self, X, y):
        '''
        Builds a tree with the given data

        X   - the input data/features 
        y   - the target values
        '''
        #self.X = X
        #self.y = y
        self.classes = np.unique(y)

        self.tree = Node()
        self.tree.depth = 0

        self.build_tree(X, y, self.tree)




    def predict(self, X_test):
        '''
        Finds the predicted class of the data in X_test using
        the built decision tree

        X_test  - the test data to make prediciton for
        '''
        if not self.tree:
            #throw error
            return None
        pass

    def predict_probabilities(self, X_test):
        '''
        Finds the predicted probabilities for the data in X_test 
        using the built decision tree.

        X_test  - the test data to make prediciton for
        '''
        if not self.tree:
            #throw error
            return None
        pass

    def build_tree(self, X, y, node):

        # if max depth not reached

        # find best split for current node
        left, right, feature, threshold = self.find_split(X,y)

        # create childe nodes
        left_node, right_node = self.create_child_nodes(node, feature, threshold)

        # continue building tree
        build_tree(X[left], y[left], left_node)
        build_tree(X[right], y[right], right_node)
        
        pass

    def create_child_nodes(node, feature, threshold):
        left_node = Node()
        left_node.depth = node.depth + 1
        right_node = Node()
        right_node.depth = node.depth + 1
        node.left = left_node
        node.right = right_node
        node.feature = feature
        node.threshold = threshold
        return left_node, right_node

    def find_split(self):
        start_impurity = self.calculate_impurity()
        split_threshold, split_feature, split_impurity
        for each feature in self.input_data:
            for each unique_value in feature:
                threshold = value
                impurity = self.calculate_impurity()
                if impurity is better than split_impurity:
                    split_threshold = threshold
                    split_impurity = impurity
                    split_feature = feature
        self.split_node(split_feature, split_threshold, self.input_data, target)

    def calculate_impurity(self):
        '''
        Calculates the impurity using the chosen impurity measure
        '''
        if(self.measure=='gini'):
            return self.gini_index()

        return

    def gini_index(self):
        '''
        Calculate the gini index/gini impurity

        gini = 1-sum(probabilities)
        '''
        pass





def gini_index_matrix(feature, input_data, target):
    '''
    Calculate the gini index for the data based on feature.

    returns the value for the gini index

    feature     - the column number of the input feature to calculate 
                  the gini index for
    input_data  - a matrix with the predictors
    target      - an array with the target/output values
    '''
    unique_values, frequency = np.unique(input_data[:,feature], return_counts=True)
    N = input_data.shape[0]
    gini = 0

    for i in range(len(unique_values)):
        value = unique_values[i]
        N_k = frequency[i]
        indexes = np.where(input_data[:,feature]==value)
        classes, n_k = np.unique(target[indexes],  return_counts=True) 
        gi = 1-1/(N_k**2)*np.sum(n_k**2)
        gini  = gini + gi*N_k
        print(gi)

    return gini/N





def gini_index_dataframe(feature, data, target):
    '''
    Calculate the gini index for the data based on feature.

    returns the value for the gini index

    feature     - the name of the input feature to calculate the 
                  gini index for
    data         - the dataframe with the data
    target      - the name of the target column in the dataframe
    '''
    unique_values = data[feature].value_counts()#.keys().tolist()
    N = data.shape[0]
    gini = 0

    for key in unique_values.keys():
        current_data = data[data[feature] == key]
        N_k = unique_values[key]
        n_k = np.array(current_data[target].value_counts().tolist())
        gi = 1-1/(N_k**2)*np.sum(n_k**2)
        gini  = gini + gi*N_k
        print(gi)

    return gini/N


def entropy():
    return 

def p_mk():
    pass

def test_run():
    df = pd.read_csv("go_for_run.csv", sep=",")
    print(gini_index_dataframe('Humidity',df,'Decision'))

    target = df['Decision'].to_numpy()
    matrix = df.drop(['Decision'], axis=1).to_numpy()

    print()
    print(df)
    print()

    print(gini_index_matrix(2, matrix, target))

if __name__ == '__main__':
    
    import seaborn as sns
    from sklearn.datasets import load_iris

    data = load_iris()
    X,y,features = data['data'], data['target'],data['feature_names']

    df = pd.DataFrame(X,columns=features)
    df['target'] = y

    # sns.pairplot(df)
    # plt.show()






