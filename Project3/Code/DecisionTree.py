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



def gini_index(feature, data, target):
    #possible_values = count unique values of feature
    unique_values = data[feature].value_counts()#.keys().tolist()
    N = data.shape[0]
    gini = 0
    n_no = 0
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

if __name__ == '__main__':
    df = pd.read_csv("go_for_run.csv", sep=",")
    print(gini_index('Outlook',df,'Decision'))

    print()
    print(df)





