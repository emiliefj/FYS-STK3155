# Exercise 1a)

# Imports
import numpy as np
import CreateData as cd
import OrdinaryLeastSquares as ols


#
# Make data and preprocess
#
n = 4 
degree = 2
test_fraction = 0.2
data = cd.CreateData(n,8)
data.add_normal_noise(0,1)
data.create_design_matrix(degree)
data.split_dataset(test_fraction)
data.scale_dataset()

#
# Perform regression
#
OLS = ols.OrdinaryLeastSquares(data.X_train_scaled,data.z_train)
OLS.regress()