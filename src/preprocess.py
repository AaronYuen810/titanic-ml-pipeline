from sklearn.base import BaseEstimator, TransformerMixin

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Stateful preprocessor for the Titanic dataset.
    Learns the median Age during fitting and transforms features into numeric format
    expected by the model.
    """
    def __init__(self):
        self.age_median = None

    def fit(self, X, y=None):
        # Learn median age from the training set
        if 'Age' in X.columns:
            self.age_median = X['Age'].median()
        return self

    def transform(self, X):
        # X is expected to be a pandas DataFrame
        X_out = X.copy()
        
        # 1. Age Imputation
        if self.age_median is not None and 'Age' in X_out.columns:
            X_out['Age'] = X_out['Age'].fillna(self.age_median)
        elif 'Age' in X_out.columns and X_out['Age'].isna().any():
            # Fallback if fitted median is somehow unavailable
            X_out['Age'] = X_out['Age'].fillna(28.0)
            
        # 2. Create Age_Binary_adult
        if 'Age' in X_out.columns:
            X_out['Age_Binary_adult'] = (X_out['Age'] > 16).astype(int)
            
        # 3. Create Sex_male
        if 'Sex' in X_out.columns:
            X_out['Sex_male'] = (X_out['Sex'] == 'male').astype(int)
            
        # 4. Subset to final expected features
        expected_columns = ['Pclass', 'Sex_male', 'Age_Binary_adult']
        
        # Ensure only expected columns exist
        for col in list(X_out.columns):
            if col not in expected_columns:
                X_out = X_out.drop(columns=[col])
                
        # Handle case if an expected column is totally missing from input
        for col in expected_columns:
            if col not in X_out.columns:
                X_out[col] = 0

        # Return DataFrame strictly in the expected feature order
        return X_out[expected_columns]
