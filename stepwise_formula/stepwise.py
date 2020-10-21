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
    
    #X = dataframe.drop(columns=[y_name])
    X = dataframe[map(lambda x: x.__repr__() ,f.terms()]
    y = dataframe[y_name]

    return backwardSelection(X,y, model_type=model_type)