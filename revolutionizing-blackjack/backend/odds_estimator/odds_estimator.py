import pandas as pd
from xgboost import XGBClassifier

from params import *


def load_odd_estimator_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(ODDS_MODEL_PATH)

    print("✅ XGBoost model for odds estimation loaded succefully")
    return model


def predict_odd_estimator_model(model: XGBClassifier, input: pd.DataFrame):
    odds = model.predict_proba(input)
    print("✅ Winning odds estimated succefully")
    return odds


model = load_odd_estimator_model()
sample_test_data = pd.DataFrame(
    {
        "player_card_1": [10],  # Player's first card
        "player_card_2": [11],  # Player's second card
        "player_card_3": [0],  # Player's third card
        "player_card_4": [0],  # Player's fourth card
        "player_card_5": [0],  # Player's fifth card
        "player_card_6": [0],  # Player's sixth card
        "player_card_7": [0],  # Player's seventh card
        "dealer_card_1": [2],  # Dealer's first card
        "dealer_card_2": [0],  # Dealer's second card
        "dealer_card_3": [0],  # Dealer's third cad
        "dealer_card_4": [0],  # Dealer's fourth card
        "dealer_card_5": [0],  # Dealer's fifth card
        "dealer_card_6": [0],  # Dealer's sixth card
        "dealer_card_7": [0],  # Dealer's seventh card
        "action_taken_1_D": [0],  # Action taken on first hand (Dealer)
        "action_taken_1_H": [0],  # Action taken on first hand (Hit)
        "action_taken_1_N": [0],  # Action taken on first hand (Double)
        "action_taken_1_P": [0],  # Action taken on first hand (Split)
        "action_taken_1_R": [0],  # Action taken on first hand (Surrender)
        "action_taken_1_S": [1],  # Action taken on first hand (Stand)
        "action_taken_2_D": [0],  # Action taken on second hand (Dealer)
        "action_taken_2_H": [0],  # Action taken on second hand (Hit)
        "action_taken_2_P": [0],  # Action taken on second hand (Split)
        "action_taken_2_R": [0],  # Action taken on second hand (Surrender)
        "action_taken_2_S": [0],  # Action taken on second hand (Stand)
        "action_taken_2_None": [1],  # Action taken on second hand
        "action_taken_3_D": [0],  # Action taken on third hand (Dealer)
        "action_taken_3_H": [0],  # Action taken on third hand (Hit)
        "action_taken_3_P": [0],  # Action taken on third hand (Split)
        "action_taken_3_S": [0],  # Action taken on third hand (Stand)
        "action_taken_3_None": [1],  # Action taken on third hand
        "action_taken_4_D": [0],  # Action taken on fourth hand (Dealer)
        "action_taken_4_H": [0],  # Action taken on fourth hand (Hit)
        "action_taken_4_S": [0],  # Action taken on fourth hand (Stand)
        "action_taken_4_None": [1],  # Action taken on fourth hand
        "action_taken_5_H": [0],  # Action taken on fifth hand (Hit)
        "action_taken_5_S": [0],  # Action taken on fifth hand (Stand)
        "action_taken_5_None": [1],  # Action taken on fifth hand
        "action_taken_6_H": [0],  # Action taken on sixth hand (Hit)
        "action_taken_6_S": [0],  # Action taken on sixth hand (Stand)
        "action_taken_6_None": [1],  # Action taken on sixth hand
    }
)
prediction = predict_odd_estimator_model(model, sample_test_data)
print(prediction)
