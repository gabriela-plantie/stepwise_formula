from .stepwiseSelector import backwardSelection
from .formula import Formula
from io import StringIO
import statsmodels.formula.api as smf
import pandas as pd


def stepwise(formula, dataframe, model_type, elimination_criteria='aic', sl= 0.05, verbose = False):
    import sys, os
    
    if verbose == False:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    try:
        formula = formula.replace('\n', ' ')
        formula = formula.replace(' ', '')
        splitted = formula.split('~')
        y_name = splitted[0]
        formula = splitted[1]
        f = Formula.fromString(formula)
        print(f.terms())
        dataframe = dataframe.copy()

        # add terms of formula as variables in data frame
        for term in filter(lambda x: isinstance(x, Formula), f.terms()):
            dataframe[term.__repr__()] = term.apply(dataframe)

        usedFields = map(lambda x: x.__repr__() if isinstance(x, Formula) else x, f.terms())
        X = dataframe[usedFields]
        y = dataframe[y_name]

        backwardModel = backwardSelection(X, y, model_type = model_type, elimination_criteria=elimination_criteria,  sl=sl)

        model_variables = _get_model_variables_dataframe(backwardModel)
        model_variables = _clean_model_variables_dataframe(model_variables, sl, dataframe)

        list_variables = [x.strip().replace('*', ':') for x in list(model_variables.variable) if x != 'intercept' ]

        formula = 'e ~ ' + ' + '.join(list_variables)

        dataframe.columns = [x.strip().replace('*', ':') for x in list(dataframe.columns)]
        X = dataframe[list_variables]
        backwardModel = backwardSelection(X, y, model_type = model_type, elimination_criteria=elimination_criteria,  sl=sl)
        
    finally:
        if verbose == False:
            sys.stdout = old_stdout

    return Model(backwardModel[2], model_type)


def _get_model_variables_dataframe(backwardModel):
    model_variables = backwardModel[2].summary()
    model_variables = model_variables.tables[1]
    model_variables = pd.read_csv(StringIO(model_variables.as_csv()) )
    model_variables.columns = ['variable','coef', 'std_error', 't', 'P_value', 'ci_0025','ci_0975' ]
    
    model_variables['variable'] = model_variables['variable'].apply(lambda x: x.strip())
    model_variables['coef'] = model_variables['coef'].astype(float)
    model_variables['std_error'] = model_variables['std_error'].astype(float)
    model_variables['t'] = model_variables['t'].astype(float)
    model_variables['P_value'] = model_variables['P_value'].astype(float)
    model_variables['ci_0025'] = model_variables['ci_0025'].astype(float)
    model_variables['ci_0975'] = model_variables['ci_0975'].astype(float)
    return model_variables


def _clean_model_variables_dataframe(model_variables, sl, dataframe):
    #remove model variables where P_value < sl
    model_variables = model_variables[model_variables['P_value']<sl]
    #q no  incluya el 0
    summ_filtered = model_variables[((model_variables.ci_0025 <=0) & (model_variables.ci_0975>=0)) == False]

    #valor del coef (consider variable range)
    filter_coef = 0.000001
    variables_with_small_coef = list(summ_filtered.variable[(abs(summ_filtered.coef) < filter_coef)])
    
    variables_with_small_coef = [x for x in variables_with_small_coef if x != 'intercept' ]
    
    insignificant_variables = []
    for variable_name in variables_with_small_coef:
        variable_range = abs(max(dataframe[variable_name]) - min(dataframe[variable_name]))
        variable_coef = float(summ_filtered.coef[summ_filtered.variable == variable_name])
        if (variable_range * variable_coef) < 0.0001:
            insignificant_variables.append(variable_name)
    
    summ_filtered = summ_filtered[summ_filtered.variable.isin(insignificant_variables) == False]
    
    return summ_filtered
    

class Model:
    def __init__(self, model, model_type):
        self.model = model
        self.params= list(model.params.index)
        final_vars = ' + '.join(self.params)
        self._formula = Formula.fromString(final_vars)
        self.model_type = model_type

    def predict(self, dataframe, alpha):
        #print('prediction')
        dataframe = dataframe.copy()
        for term in filter(lambda x: isinstance(x, Formula), self._formula.terms()):
            dataframe[term.__repr__()] = term.apply(dataframe)
    
        #usedFields = list(map(lambda x: x.__repr__() if isinstance(x, Formula) else x, self._formula.terms()))
        intercepts = [x for x in self.params if x =='intercept' or x == 'Intercept']
        for intercept in intercepts:
            dataframe[intercept] = 1

        
        X = dataframe[map(lambda x: x.replace(':', '*'), self.params)] #se rompe si no tiene el orden
                                    #se rompe cuando hay intercepto 

        if self.model_type=='linear':
            tbl_prediction = self.model.get_prediction(X)
            intervals = tbl_prediction.summary_frame(alpha= alpha)
            prediction= intervals['mean']
        else:
            prediction = self.model.predict(X)
            intervals = None
            
        return TestResults(X, list(prediction), intervals)

        



class TestResults:
    def __init__(self, testDataframe, prediction, intervals = None):
        self.testDataframe = testDataframe
        self.prediction= prediction
        self.intervals = intervals

