from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from classification_model.config.core import config
from classification_model.processing import features as pp
from feature_engine.encoding import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

Titanic = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
                fill_value= "missing",
                ignore_format=True,
            ),
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
        ),
        # impute numerical variables with the mean
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),  
        # Extract first letter from cabin
    ('extract_letter', pp.ExtractLetterTransformer(variables=config.model_config.cabin_variable_with_imputation)),
    
        # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars, replace_with="RARE", ignore_format=True,
            ),
        ),
        # encode categorical variables using the target mean
        (
            "categorical_encoder",
            OneHotEncoder(
                variables=config.model_config.categorical_vars, drop_last=True
            ),
        ),
        ("scaler", StandardScaler()),
        # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=0.0005, random_state=0)),
    ]
)
