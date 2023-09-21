import pandas as pd
from xgboost import XGBClassifier

from params import *


def load_odd_estimator_model() -> XGBClassifier:
    """
    Load an XGBoost model for odds estimation from a pre-trained model file.

    Returns:
        XGBClassifier: Loaded XGBoost model for odds estimation.
    """
    model = XGBClassifier()
    model.load_model(ODDS_MODEL_PATH)

    print("✅ XGBoost model for odds estimation loaded succefully")
    return model


def predict_odd_estimator_model(model: XGBClassifier, input: dict) -> list:
    """
    Predict the winning odds using a loaded XGBoost model.

    Args:
        model (XGBClassifier): Loaded XGBoost model for odds estimation.
        input (dict): Dictionary containing input features for prediction.

    Returns:
        list: A list of winning odds estimated for the input data.
    """
    input_df = pd.DataFrame(input)
    odds = model.predict_proba(input_df)
    odds = odds[0]
    odds = odds.tolist()
    print("✅ Winning odds estimated succefully")
    return odds
