from .stepwiseSelector import backwardSelection
from .formula import Formula

def stepwise(formula, dataframe, model_type):
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

    backwardModel = backwardSelection(X,y, model_type=model_type)
    return Model(backwardModel[2])


class Model:
    def __init__(self, model):
        self.model = model
        final_vars = ' + '.join(list(model.params.index[1:]))
        self._formula = Formula.fromString(final_vars)

    def predict(self, dataframe):
        dataframe = dataframe.copy()
        for term in filter(lambda x: isinstance(x, Formula), self._formula.terms()):
            dataframe[term.__repr__()] = term.apply(dataframe)
    
        usedFields = map(lambda x: x.__repr__() if isinstance(x, Formula) else x, self._formula.terms())
        X = dataframe[usedFields]
        print(X.columns)
        return self.model.predict(X) 