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
        self.z_mesh = np.array([[0.76642059, 0.81885368, 0.43491424, 0.25206195, 0.10755755],
                                [0.80258259, 1.16528332, 0.53811211, 0.58935857, 0.23021761],
                                [0.48180615, 0.50456938, 0.32576209, 0.40804792, 0.16102556],
                                [0.33952742, 0.27241325, 0.04371943, 0.1159698,  0.05036027],
                                [0.27033716, 0.22224008, 0.14597916, 0.08104742, 0.03586959]])

    def test_size_generated_values(self):
        self.assertEqual(self.data.x_mesh.shape,(self.n,self.n),'x_mesh in CreateData has wrong shape')
        self.assertEqual(self.data.y_mesh.shape,(self.n,self.n),'y_mesh in CreateData has wrong shape')
        self.assertEqual(self.data.z_mesh.shape,(self.n,self.n),'z_mesh in CreateData has wrong shape')

    def test_x_y_values(self):
        self.assertTrue((self.data.x_mesh>=0).all() and (self.data.x_mesh<=1).all(),'generated x-values outside range (0,1)')
        self.assertTrue((self.data.y_mesh>=0).all() and (self.data.y_mesh<=1).all(),'generated y-values outside range (0,1)')

    def test_z_values(self):
        #new_data = CreateData(5)
        x = np.linspace(0,1,5)
        y = np.linspace(0,1,5)
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_mesh = self.data.calculate_values(x_mesh,y_mesh)
        self.assertTrue(np.allclose(z_mesh,self.z_mesh),'wrong values calculated for Franke function')

    def test_add_noise(self):
        z_no_noise = self.data.z_mesh
        self.data.add_normal_noise(0,0.01)
        z_with_noise = self.data.z_mesh
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
        
# if __name__ == "__main__":
#     unittest.main()