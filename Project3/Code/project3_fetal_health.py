# imports:
import numpy as np 
import pandas as pd

# Own code
import DecisionTree as dt
import LogisticRegression as lr
import preprocess_data as pp
import feature_selection as fs

if __name__ == '__main__':
    
    # Pre-process data #

    df = pp.read_data('../Data/Fetal Health/fetal_health.csv', names=None)
    features = np.array(df.columns[:-1])
    y = np.array(df['fetal_health'])
    X = np.array(df.drop(['fetal_health'], axis=1))
    
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_train_val_test(X[1:],y[1:],0.6,0.2,0.2)



    # Feature selection # 