from .stepwiseSelector import backwardSelection
from .formula import Formula

def stepwise(formula, dataframe, model_type, elimination_criteria='aic', sl=sl):
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

    backwardModel = backwardSelection(X,y, model_type=model_type,elimination_criteria=elimination_criteria, sl=sl )
    return Model(backwardModel[2])


class Model:
    def __init__(self, model):
        self.model = model
        self.params= list(model.params.index)
        final_vars = ' + '.join(self.params)
        self._formula = Formula.fromString(final_vars)

    def predict(self, dataframe):
        print('prediction')
        dataframe = dataframe.copy()
        for term in filter(lambda x: isinstance(x, Formula), self._formula.terms()):
            dataframe[term.__repr__()] = term.apply(dataframe)
    
        usedFields = list(map(lambda x: x.__repr__() if isinstance(x, Formula) else x, self._formula.terms()))
        
        if 'intercept' in self.params:
            dataframe['intercept'] = 1

        X = dataframe[self.params] #se rompe si no tiene el orden
                                    #se rompe cuando hay intercepto 
        print(X.columns)
        #return X
        return TestResults(X, self.model.predict(X))



class TestResults:
    def __init__(self, testDataframe, prediction):
        self.testDataframe = testDataframe
        self.prediction= prediction

