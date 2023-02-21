from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        self.var = variables


    def fit(self, X, y = None):
        return self
        

    def transform(self, X, y = None):
        output = X.copy()
        for var in self.var:
            output[var] = output[var].str.extract('([A-Za-z]+)', expand=False)
        return output
        