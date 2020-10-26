import unittest

import pandas as pd
import numpy as np
import random
import scipy as sc

from stepwise_formula.stepwise import stepwise

#from matplotlib import pyplot as plt
#import statsmodels.formula.api as smf
#import statsmodels.api as sm

class TestStepwise(unittest.TestCase):

    def test_stepwise_logistic(self):
        x1=np.random.normal(0,1,1000)
        x2=np.random.normal(0,1,1000)
        x3=np.random.normal(0,1,1000)
        x4=np.random.normal(0,1,1000)/5

        z=4+ 2*x1 + x2 + 2*x2*x3 
        pr=1/(1+np.exp(-z))
        y=sc.stats.binom.rvs(1, pr, size=1000)

        df = pd.DataFrame(data={'y':y, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4})

        formula = 'y ~ x1 + x2 * x3'
        model = stepwise(formula, df, 'logistic', verbose=False)

        params = model.model.params

        tolerance = 1

        self.assertTrue( abs(params['intercept'] - 4) < tolerance)
        self.assertTrue( abs(params['x1'] - 2) < tolerance)
        self.assertTrue( abs(params['x2'] - 1) < tolerance)
        self.assertTrue( abs(params['x2*x3'] - 2) < tolerance)

    def test_stepwise_lineal(self):
        x1=np.random.normal(0,1,1000)
        x2=np.random.normal(0,1,1000)
        x3=np.random.normal(0,1,1000)
        x4=np.random.normal(0,1,1000)/5

        y= 4+ 2*x1 + x2 + 2*x2*x3 

        df = pd.DataFrame(data={'y':y,'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4})

        formula = 'y ~ x1 + x2 * x3'
        model = stepwise(formula, df, 'linear', verbose=False)

        params = model.model.params

        tolerance = 1

        self.assertTrue( abs(params['intercept'] - 4) < tolerance)
        self.assertTrue( abs(params['x1'] - 2) < tolerance)
        self.assertTrue( abs(params['x2'] - 1) < tolerance)
        self.assertTrue( abs(params['x2*x3'] - 2) < tolerance)

if __name__ == '__main__':
    unittest.main()