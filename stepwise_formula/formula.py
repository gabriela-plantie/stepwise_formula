class Formula:
    
    class Operator:
        
        def __init__(self, symbol, function):
            self.symbol = symbol
            self.apply = function
        
    multiplication = Operator('*', lambda a, b: a * b)
    multiplication2 = Operator(':', lambda a, b: a * b)
    add = Operator('+', lambda a, b: a + b)

    operators = [add, multiplication, multiplication2]
    
    def __init__(self, operation, term1, term2):
        self.operation = operation
        self.term1 = term1
        self.term2 = term2
        
    def apply(self, dataframe):
        if isinstance(self.term1, Formula):
            t1 = self.term1.apply(dataframe)
        else:
            t1 = dataframe[self.term1]
            
        if isinstance(self.term2, Formula):
            t2 = self.term2.apply(dataframe)
        else:
            t2 = dataframe[self.term2]
        
        return self.operation.apply(t1, t2)
    
    @classmethod
    def fromString(cls, formulaString):
        formulaString = formulaString.replace(' ', '')
        for operation in cls.operators:
            if operation.symbol in formulaString:
                terms = formulaString.split(operation.symbol, 1)
                return Formula(operation, Formula.fromString(terms[0]), Formula.fromString(terms[1]))
        return formulaString
    
    def terms(self):
        terms = set()
        
        if self.operation.symbol == '*' or self.operation.symbol == ':':
            
            if isinstance(self.term1, Formula):
                t1 = self.term1.terms()
            else:
                t1 = set()
                t1.add(self.term1)
                
            if isinstance(self.term2, Formula):
                t2 = self.term2.terms()
            else:
                t2 = set()
                t2.add(self.term2)
            
            terms = terms.union(t1)
            terms = terms.union(t2)

            terms = terms.union([Formula(Formula.multiplication, x1, x2) for x1 in t1 for x2 in t2])

            if self.operation.symbol == ':':
                return terms
        
        if isinstance(self.term1, Formula):
            terms = terms.union(self.term1.terms())
        else:
            terms.add(self.term1)
            
        if isinstance(self.term2, Formula):
            terms = terms.union(self.term2.terms())
        else:
            terms.add(self.term2)
        
        return terms
    
    def simpleTerms(self):
        terms = set()
        
        if isinstance(self.term1, Formula):
            terms = terms.union(self.term1.simpleTerms())
        else:
            terms.add(self.term1)
            
        if isinstance(self.term2, Formula):
            terms = terms.union(self.term2.simpleTerms())
        else:
            terms.add(self.term2)
        
        return terms
        
    def __repr__(self):
        return str(self.term1) + self.operation.symbol + str(self.term2)
        
    def __eq__(self, other):
        return (self.term1 == other.term1 and self.term2 == other.term2 and self.operation == self.operation) or (self.term1 == other.term2 and self.term2 == other.term1 and self.operation == self.operation)
    
    def __hash__(self):
        return hash(list(self.simpleTerms()).sort())