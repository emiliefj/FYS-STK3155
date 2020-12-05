import numpy as np 
import pandas as pd

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
        # print(gi)

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
        # print(gi)

    return gini/N


def entropy():
    return 

def p_mk():
    pass

if __name__ == '__main__':
    df = pd.read_csv("go_for_run.csv", sep=",")
    print(gini_index_dataframe('Outlook',df,'Decision'))

    target = df['Decision'].to_numpy()
    matrix = df.drop(['Decision'], axis=1).to_numpy()

    print()
    print(df)
    print()

    print(gini_index_matrix(0, matrix, target))






