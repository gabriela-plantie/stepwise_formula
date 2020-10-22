## What it does:

This packages receives a data frame and function that could have multiplicative terms and runs setpwise selection from this package.

https://github.com/talhahascelik/python_stepwiseSelection

## Install package


```python
%%capture
!pip3 install git+git://github.com/gabriela-plantie/stepwise_formula
```


```python
from stepwise_formula.stepwise import stepwise
```


```python
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import random
```

## create table with simulated data


```python

def create_df(seed):
    np.random.seed(seed)
    x1=np.random.normal(0,1,1000)
    x2=np.random.normal(0,1,1000)
    x3=np.random.normal(0,1,1000)
    x4=np.random.normal(0,1,1000)/5

    z=1 + 2*x1 + x2 + 2*x2*x3 +x4
    pr=1/(1+np.exp(-z))
    y=sc.stats.binom.rvs(1, pr, size=1000)
    df = pd.DataFrame(data={'y':y, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4})
    return [df,z]
```


```python
df= create_df(30)[0]
z= create_df(30)[1]
```

## Define formula


```python
formula = 'y ~  x2*x3 + x1:x4 '
```

## Run stepwise using formula


```python
model = stepwise(formula, df, 'logistic')
```

    {'x3', x2*x3, 'x2', x1*x4}
    Character Variables (Dummies Generated, First Dummies Dropped): []
    Optimization terminated successfully.
             Current function value: 0.541247
             Iterations 6
    Eliminated : x3
    Optimization terminated successfully.
             Current function value: 0.541255
             Iterations 6
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1000
    Model:                          Logit   Df Residuals:                      996
    Method:                           MLE   Df Model:                            3
    Date:                Thu, 22 Oct 2020   Pseudo R-squ.:                  0.1806
    Time:                        15:38:16   Log-Likelihood:                -541.26
    converged:                       True   LL-Null:                       -660.53
    Covariance Type:            nonrobust   LLR p-value:                 1.954e-51
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.6208      0.075      8.286      0.000       0.474       0.768
    x2*x3          1.3555      0.122     11.097      0.000       1.116       1.595
    x2             0.6711      0.093      7.240      0.000       0.489       0.853
    x1*x4          1.3253      0.371      3.574      0.000       0.599       2.052
    ==============================================================================
    AIC: 1090.51050046908
    BIC: 1110.1415215850084
    Final Variables: ['intercept', 'x2*x3', 'x2', 'x1*x4']


## Predicction on another dataset


```python
t=testResults.testDataframe
```


```python
t['pred']= model.model.predict(t[model.model.params.index])
```


```python
t
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>x2*x3</th>
      <th>x2</th>
      <th>x1*x4</th>
      <th>ord</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.139584</td>
      <td>0.452175</td>
      <td>-0.142610</td>
      <td>0.715952</td>
      <td>0.715952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.830760</td>
      <td>1.134899</td>
      <td>-0.016467</td>
      <td>0.558363</td>
      <td>0.558363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-0.046015</td>
      <td>0.077442</td>
      <td>-0.178077</td>
      <td>0.592518</td>
      <td>0.592518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-0.494669</td>
      <td>0.226821</td>
      <td>0.098642</td>
      <td>0.558039</td>
      <td>0.558039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.189129</td>
      <td>0.526648</td>
      <td>-0.051086</td>
      <td>0.761861</td>
      <td>0.761861</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1</td>
      <td>-0.164495</td>
      <td>0.897219</td>
      <td>-0.107829</td>
      <td>0.702047</td>
      <td>0.702047</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>-0.256417</td>
      <td>-0.577262</td>
      <td>-0.017205</td>
      <td>0.465798</td>
      <td>0.465798</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1</td>
      <td>0.125015</td>
      <td>0.904900</td>
      <td>-0.002307</td>
      <td>0.801311</td>
      <td>0.801311</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1</td>
      <td>0.405615</td>
      <td>1.120783</td>
      <td>-0.008264</td>
      <td>0.871228</td>
      <td>0.871228</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1</td>
      <td>-0.276153</td>
      <td>-0.420349</td>
      <td>0.077656</td>
      <td>0.516808</td>
      <td>0.516808</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 6 columns</p>
</div>



## Plot model logit vs original logit


```python
plt.scatter(list(map(lambda p: np.log(p/(1-p)), t['pred'])) , z)
```




    <matplotlib.collections.PathCollection at 0x7f6204b027c0>




    
![png](README_files/README_18_1.png)
    



```python
#jupyter nbconvert README.ipynb --to markdown
```
