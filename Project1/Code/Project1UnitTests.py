import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from CreateData import *
from OrdinaryLeastSquares import *

class CreateDataTest(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.data = CreateData(self.n)

        self.z = np.array([0.76642059, 1.16528332, 0.32576209, 0.1159698 , 0.03586959])

    def test_size_generated_values(self):
        self.assertEqual(np.size(self.data.x),self.n,'x in CreateData has wrong size')
        self.assertEqual(np.size(self.data.y),self.n,'y in CreateData has wrong size')
        self.assertEqual(np.size(self.data.z),self.n,'z in CreateData has wrong size')

    def test_z_values(self):
        #new_data = CreateData(5)
        x = np.linspace(0,1,5)
        y = np.linspace(0,1,5)
        z = self.data.calculate_values(x,y)
        self.assertTrue(np.allclose(z,self.z),'wrong values calculated for Franke function')

    def test_add_noise(self):
        z_no_noise = self.data.z
        self.data.add_normal_noise(0,0.01)
        z_with_noise = self.data.z
        self.assertFalse(np.allclose(z_no_noise,z_with_noise),'arrays are equal after adding noise')
        self.assertTrue(np.allclose(z_no_noise,z_with_noise,atol=0.05),'arrays too dissimilar after adding noise')
        

class OrdinaryLeastSquaresTest(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.data = CreateData(self.n)
        self.data.create_design_matrix(2)
        self.ols = OrdinaryLeastSquares(self.data.X, self.data.z)

    def test_init(self):
        self.assertEqual(self.ols.X.all(),self.data.X.all(), 'error in stored design matrix')
        self.assertEqual(self.ols.z.all(),self.data.z.all(), 'error in stored z values')
        self.assertFalse(self.ols.is_regressed, 'is_regressed not set to False at initialization')

    def test_regression(self):
        self.ols.fit()
        lin_model = LinearRegression() # OLS
        lin_model.fit(self.data.X, self.data.z)
        ztilde_skl = lin_model.predict(self.data.X)
        self.assertTrue(np.allclose(self.ols.ztilde,ztilde_skl), 'ols regression fit result differs from sklearn')
        self.assertTrue(self.ols.is_regressed, 'is_regressed not set to True after .fit() is called')

        new_data = CreateData(self.n)
        new_data.create_design_matrix(2)
        z_predict = self.ols.predict(new_data.X)
        z_predict_skl = lin_model.predict(new_data.X)
        self.assertTrue(np.allclose(z_predict,z_predict_skl), 'ols regression predict result differs from sklearn')
    
    def test_error_measures(self):
        self.ols.fit()
        ols_mean = self.ols.mean(self.ols.z)
        ols_r2 = self.ols.r2(self.ols.ztilde,self.ols.z)
        ols_mse = self.ols.mean_square_error()

        mean = np.mean(self.ols.z)
        r2 = r2_score(self.ols.ztilde,self.ols.z)
        mse = mean_squared_error(self.ols.ztilde,self.ols.z)

        self.assertEqual(ols_mean,mean,'mean differs from numpy value')
        self.assertEqual(ols_r2,r2,'r2 score differs from sklearn value')
        self.assertEqual(ols_mse,mse,'mse differs from sklearn value')
        
