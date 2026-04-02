from src.preprocess import TitanicPreprocessor
import pandas as pd 

def test_titanic_preprocessor_age_imputation():
    """Test that the TitanicPreprocessor can impute missing age values."""
    preprocessor = TitanicPreprocessor()
    df = pd.DataFrame({
        "Pclass": [1, 2, 3],
        "Age": [10.0, None, 30.0],
        "Sex": ["male", "female", "male"],
    })

    transformed_df = preprocessor.fit_transform(df)
    assert preprocessor.age_median == 20.0
    assert transformed_df["Age_Binary_adult"].notna().all()
    # The missing Age is imputed to 20.0 (>16), so the row is classified as adult.
    assert transformed_df["Age_Binary_adult"].tolist() == [0, 1, 1]
    assert transformed_df.columns.tolist() == ["Pclass", "Sex_male", "Age_Binary_adult"]

def test_titanic_preprocessor_sex_ohe():
    """Test that the TitanicPreprocessor can one-hot encode the sex column.
    Male should be encoded as one, female should be encoded as zero.
    """
    preprocessor = TitanicPreprocessor()
    df = pd.DataFrame({
        "Pclass": [1, 2, 3, 3],
        "Age": [20.0, 20.0, 20.0, 20.0],
        "Sex": ["MALE", "female", "Male", "FEMALE"],
    })

    transformed_df = preprocessor.fit_transform(df)
    assert transformed_df["Sex_male"].notna().all()

    assert transformed_df["Sex_male"].tolist() == [1, 0, 1, 0]
    assert transformed_df.columns.tolist() == ["Pclass", "Sex_male", "Age_Binary_adult"]
