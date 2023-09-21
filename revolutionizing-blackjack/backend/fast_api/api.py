from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import os
import re

# Import functions and classes from project modules
from params import *

from computer_vision.computer_vision import (
    load_roboflow_model,
    predict_roboflow_model,
    load_model,
    image_decoder,
    get_predictions,
)

from move_recommender.move_recommender import (
    Hand,
    SCORE_TABLE,
    check_winner,
    EX,
)

from odds_estimator.odds_estimator import (
    load_odd_estimator_model,
    predict_odd_estimator_model,
)

app = FastAPI()

# Allow all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Cache models at API startup
app.state.roboflow_model = load_roboflow_model()
app.state.model = load_model(MODEL_PATH)
app.state.xgboost_model = load_odd_estimator_model()


@app.get("/")
def index():
    """
    Endpoint for testing the API status.
    """
    return {"status": "ok"}


@app.post("/card_predictions_roboflow")
async def receive_image(img: UploadFile = File(...)):
    """
    Given image, returns predictions and clusters from RoboFlow model.
    Returns None if there are no predictions
    """
    # Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    # Image directory and file name
    directory = os.path.join("computer_vision", "temp_image")
    filename = "input.png"

    # Temporarly saves image
    cv2.imwrite(os.path.join(directory, filename), cv2_img)

    # Call roboflow model function
    predictions = predict_roboflow_model(app.state.roboflow_model)

    # delete the temp image
    os.remove(os.path.join(directory, filename))

    if predictions is None:
        detections = 0
        classes = []
        bounding_boxes = []
        confidence = []

    else:
        detections = predictions.shape[0]
        classes = predictions["class"].tolist()
        bounding_boxes = []
        for index, row in predictions.iterrows():
            bounding_boxes.append([row["x"], row["y"], row["width"], row["height"]])
        confidence = predictions["confidence"].tolist()

    return {
        "detections": detections,
        "cards detected": classes,
        "bounding boxes": bounding_boxes,
        "confidence": confidence,
    }


@app.post("/card_predictions_yolo")
async def receive_image(img: UploadFile = File(...)):
    """
    Given image, returns predictions and clusters from own trained model.
    Returns None if there are no predictions
    """
    contents = await img.read()
    decoded_img = image_decoder(contents)
    predictions = get_predictions(model=app.state.model, decoded_img=decoded_img)

    return predictions


@app.post("/predict_move")
def predict(input: dict = {"dealer": ["10H"], "player": ["2D", "4C"]}):
    """
    Given a dictionary with the dealer and player cards, it returns the current status of the gameplay
    and recommends the best move to the player.

    Args:
        input (dict): A dictionary containing "dealer" and "player" lists of cards.

    Returns:
        dict: A dictionary with information about the next move, player's hand score, dealer's hand score,
              and a message.
    """
    # Extract player and dealer cards from the input dictionary
    player_cards = [re.sub("[SCDH]", "", i) for i in input.get("player")]
    dealer_cards = [re.sub("[SCDH]", "", i) for i in input.get("dealer")]

    # Create player and dealer hand objects
    player_hand = Hand(player_cards)
    dealer_hand = Hand(dealer_cards)

    # Check if the dealer has more than one card to determine the game status
    if len(dealer_cards) > 1:
        # Check the winner of the game
        message = check_winner(player_hand, dealer_hand)

        # Get player's and dealer's hand scores
        player_score = player_hand.get_score()

        # Return the game status and information
        return {
            "next_move": "Dealer's turn.",
            "player_hand": player_score,
            "dealer_hand": dealer_hand.get_score(),
            "message": message,
        }
    else:
        # Get player's hand score
        player_score = player_hand.get_score()

        if player_hand.get_score() > 21:
            rec = "Player is busted."

        elif player_hand.is_blackjack():
            rec = "Player got a Blackjack."

        else:
            rec = EX.get(player_hand.recommend(dealer_hand))

        # Return the recommended move and player's hand score
        return {
            "next_move": rec,
            "player_hand": player_score,
            "dealer_hand": dealer_hand.get_score(),
            "message": "",
        }


@app.post("/odds_estimator")
def predict(
    input: dict = {
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
):
    """
    Predicts the odds of winning and losing based on input data.

    Args:
        input (dict): A dictionary containing current situation of the game.
                      The keys represent feature names, and the values are lists of
                      feature values.

    Returns:
        dict: A dictionary containing the predicted odds of winning and losing.
              The keys are "win" and "lose," and the values are the predicted
              probabilities of winning and losing, respectively.
    """
    # Call the predict_odd_estimator_model function to make predictions
    prediction = predict_odd_estimator_model(app.state.xgboost_model, input)
    # Return the predicted odds
    return {"win": prediction[1], "lose": prediction[0]}
