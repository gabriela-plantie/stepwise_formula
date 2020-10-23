from .stepwiseSelector import backwardSelection
from .formula import Formula


def stepwise(formula, dataframe, model_type, elimination_criteria='aic', sl= 0.05, verbose = True):
    import sys, os
    if verbose == False:
        sys.stdout = open(os.devnull, 'w')
    
    formula = formula.replace('\n', ' ')
    formula = formula.replace(' ', '')
    splitted = formula.split('~')
    y_name = splitted[0]
    formula = splitted[1]
    f = Formula.fromString(formula)
    print(f.terms())
    dataframe = dataframe.copy()
    for term in filter(lambda x: isinstance(x, Formula), f.terms()):
        dataframe[term.__repr__()] = term.apply(dataframe)
    
    usedFields = map(lambda x: x.__repr__() if isinstance(x, Formula) else x, f.terms())
    X = dataframe[usedFields]
    y = dataframe[y_name]

    backwardModel = backwardSelection(X,y, model_type=model_type,elimination_criteria=elimination_criteria, sl=sl)

    if verbose == False:
        sys.stdout = sys.__stdout__

    return Model(backwardModel[2], model_type)


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
        
        if 'intercept' in self.params:
            dataframe['intercept'] = 1

        X = dataframe[self.params] #se rompe si no tiene el orden
                                    #se rompe cuando hay intercepto 

        if self.model_type=='linear':
            tbl_prediction = self.model.get_prediction(X)
            intervals = tbl_prediction.summary_frame(alpha= alpha)
            prediction= intervals['mean']
        else:
            prediction = self.model.predict(X)
            intervals = None
            
        return TestResults(X, prediction, intervals)

        



class TestResults:
    def __init__(self, testDataframe, prediction, intervals = None):
        self.testDataframe = testDataframe
        self.prediction= prediction
        self.intervals = intervals

