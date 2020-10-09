## What it does:

This package receives a data frame and function that could have multiplicative terms and runs setpwise selection from this package.

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
```

## create table with simulated data


```python
x1=np.random.normal(0,1,1000)
x2=np.random.normal(0,1,1000)
x3=np.random.normal(0,1,1000)
x4=np.random.normal(0,1,1000)/5

z=1 + 2*x1 + x2 + 2*x2*x3 + x4
pr=1/(1+np.exp(-z))
y=sc.stats.binom.rvs(1, pr, size=1000)
df = pd.DataFrame(data={'y':y, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4})

```

## Define formula


```python
formula = 'y ~  x2*x3 + x1:x4 '
```

## Run stepwise using formula


```python
a = stepwise(formula, df, 'logistic')
```

    {x2*x3, x1*x4, 'x3', 'x2'}
    Character Variables (Dummies Generated, First Dummies Dropped): []
    Optimization terminated successfully.
             Current function value: 0.394587
             Iterations 7
    Eliminated : x3
    Optimization terminated successfully.
             Current function value: 0.394595
             Iterations 7
    Eliminated : x1*x4
    Optimization terminated successfully.
             Current function value: 0.396311
             Iterations 7
    Regained :  x1*x4
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1000
    Model:                          Logit   Df Residuals:                      994
    Method:                           MLE   Df Model:                            5
    Date:                Wed, 07 Oct 2020   Pseudo R-squ.:                  0.4026
    Time:                        13:48:16   Log-Likelihood:                -394.60
    converged:                       True   LL-Null:                       -660.53
    Covariance Type:            nonrobust   LLR p-value:                1.049e-112
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.9307      0.097      9.556      0.000       0.740       1.122
    x1             1.8153      0.134     13.545      0.000       1.553       2.078
    x2             0.7976      0.111      7.154      0.000       0.579       1.016
    x4             0.8838      0.449      1.968      0.049       0.004       1.764
    x2*x3          1.9254      0.165     11.660      0.000       1.602       2.249
    x1*x4         -1.0898      0.579     -1.881      0.060      -2.226       0.046
    ==============================================================================
    AIC: 801.1906769633258
    BIC: 830.6372086372187
    Final Variables: ['intercept', 'x1', 'x2', 'x4', 'x2*x3', 'x1*x4']


## filter final list of variables by pvalue


```python
np.round(a[2].params[a[2].pvalues<0.01],2)
```




    intercept    0.93
    x1           1.82
    x2           0.80
    x2*x3        1.93
    dtype: float64



## generate model with resulting varibles


```python
formula = 'y ~ x1 + x2 + x2:x3'
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
mod = model.fit()
```


```python
np.round(mod.params[mod.pvalues<0.01],2)
```




    Intercept    0.92
    x1           1.80
    x2           0.78
    x2:x3        1.89
    dtype: float64




```python
pred=mod.predict(df)
```


```python
plt.scatter(pr, pred)
```




    <matplotlib.collections.PathCollection at 0x7f18adb912b0>




    
![png](README_files/README_18_1.png)
    



```python

```


```python
#jupyter nbconvert README.ipynb --to markdown
```


```python

```
