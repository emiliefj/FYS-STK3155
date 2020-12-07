import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class DecisionTree():
    '''
    A decision tree for classification.


    '''
    def __init__(self, impurity_measure='gini', max_depth=5, min_samples_leaf=1, max_leaf_nodes=8):
        
        self.measure = impurity_measure.lower()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

        self.n_leaves = 0
        self.depth = 0

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


        print("Created a decision tree with",self.n_leaves,"leaves, and a depth of", self.depth, " at the deepest.")




    def predict(self, X_test):
        '''
        Finds the predicted class of the data in X_test using
        the built decision tree

        X_test  - the test data to make prediciton for
        '''
        if not self.tree:
            #throw error, fit first
            return None

        n_pred = X_test.shape[0]
        predictions = np.zeros(n_pred) # For storing predictions

        for i in range(n_pred):
            
            predictions[i] = np.argmax(self.make_prediction(self.tree,X_test[i]))
            # print(X_test[i], predictions[i])

        return predictions
            
    def make_prediction(self,node,entry):
        if(node.leaf): # reached leaf node, found prediction
            return node.prediction
        if(entry[node.feature]<node.threshold):
            return self.make_prediction(node.left, entry)
        else:
            return self.make_prediction(node.right, entry)



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
        values, frequency = np.unique(y,  return_counts=True)
        probabilities = np.zeros(len(self.classes))
        probabilities[values] = frequency/len(y)
        node.prediction = probabilities
        print("Making leaf node with prediction: ", node.prediction)


    def build_tree(self, X, y, node):

        if(node.depth>self.depth):
            self.depth = node.depth

        # Check stopping criteria #

        # reached max depth
        if node.depth >= self.max_depth:
            self.make_leaf(node,y)
            return
        
        # if X.shape[0] < self.min_samples_split:
        #     node.leaf = True
        #     return
        
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

    def find_split(self, X, y):
        start_impurity = self.calculate_impurity(y) # ?

        # print(pd.DataFrame(X,y))
        split_impurity = start_impurity #start_impurity

        N = X.shape[0]
        N_features = X.shape[1]

        prediction = np.mean(y)
        split_threshold = None
        split_feature = None
        right = None
        left = None

        for i in range(N_features):
            unique_values, frequency = np.unique(X[:,i], return_counts=True)

            for val in unique_values:
                threshold = val
                # if categorical: ==/!=
                left_index = np.where(X[:,i]<val)
                right_index = np.where(X[:,i]>=val)

                impurity_left = self.calculate_impurity(y[left_index])
                impurity_right = self.calculate_impurity(y[right_index])
                impurity = impurity_left + impurity_right
                if(impurity<split_impurity):
                    #print("!winner! feat: ", i, " split at: ", threshold, "  impurity=",impurity)

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

class Node():
    '''

    feature     - The feature the node makes its deciosion based on
    threshold   - The threshold separating the left and right branch of the node

    '''
    def __init__(self):#, feature, threshold):
        
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
    from IPython.display import display 
    display(list(df.columns.values)) 
    # print(display(df))
    # print(X.shape)

    tree = DecisionTree()
    #tree.fit(X,y)
    n = len(y)
    np.random.seed(13)
    indexes = np.random.randint(n,size=50) # 10

    #tree.fit(X[indexes],y[indexes])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=13)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_train)
    print(accuracy(y_pred,y_train))






