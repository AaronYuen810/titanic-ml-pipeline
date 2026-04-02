from sklearn.base import BaseEstimator, TransformerMixin


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_median = None

    def fit(self, X, y=None):
        if "Age" in X.columns:
            self.age_median = X["Age"].median()
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. Age imputation
        if self.age_median is not None and "Age" in X_copy.columns:
            X_copy["Age"] = X_copy["Age"].fillna(self.age_median)
        elif "Age" in X_copy.columns and X_copy["Age"].isna().any():
            # fall back to median age if somehow age_median state is unavailable.
            X_copy["Age"] = X_copy["Age"].fillna(28.0)

        # 2. Bin Age into young and adult
        if "Age" in X_copy.columns:
            X_copy["Age_Binary_adult"] = (X_copy["Age"] > 16).astype(int)

        # 3. Normalize Sex column
        if "Sex" in X_copy.columns:
            X_copy["Sex"] = X_copy["Sex"].str.lower()

        # 4. Create Sex_male
        if "Sex" in X_copy.columns:
            X_copy["Sex_male"] = (X_copy["Sex"] == "male").astype(int)

        # 5. Subset to final expected features
        expected_columns = ["Pclass", "Sex_male", "Age_Binary_adult"]

        # Ensure only expected columns exist
        for col in list(X_copy.columns):
            if col not in expected_columns:
                X_copy = X_copy.drop(columns=[col])

        # Handle case if an expected column is totally missing from input
        # Ideally this should never happen
        for col in expected_columns:
            if col not in X_copy.columns:
                X_copy[col] = 0

        # Return DataFrame strictly in the expected feature order
        return X_copy[expected_columns]
