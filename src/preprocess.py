from sklearn.base import BaseEstimator, TransformerMixin


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_median = None

    def fit(self, X, y=None):
        self.age_median = X["Age"].median()
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing Age values with the median
        X_copy["Age"] = X_copy["Age"].fillna(self.age_median)

        # Bin age into young and adult
        X_copy["Age_Binary"] = X_copy["Age"].apply(lambda x: "young" if x <= 16 else "adult") 

        # 

