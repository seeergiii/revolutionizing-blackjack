import pandas as pd
import numpy as np

# Define the decision matrices for double down, hard hand, and soft hand strategies
DH = np.array(
    [
        ["H"] * 10,
        ["H"] + ["Dh"] * 4 + ["H"] * 5,
        ["Dn"] * 8 + ["H"] * 2,
        ["Dn"] * 9 + ["H"],
        ["H"] * 2 + ["S"] * 3 + ["H"] * 5,
        ["S"] * 5 + ["H"] * 5,
        ["S"] * 5 + ["H"] * 5,
        ["S"] * 5 + ["H"] * 3 + ["Rh"] + ["H"],
        ["S"] * 5 + ["H"] * 2 + ["Rh"] * 3,
        ["S"] * 10,
    ]
)
DS = np.array(
    [
        ["H"] * 3 + ["Dh"] * 2 + ["H"] * 5,
        ["H"] * 3 + ["Dh"] * 2 + ["H"] * 5,
        ["H"] * 2 + ["Dh"] * 3 + ["H"] * 5,
        ["H"] * 2 + ["Dh"] * 3 + ["H"] * 5,
        ["H"] + ["Dh"] * 4 + ["H"] * 5,
        ["S"] + ["Ds"] * 4 + ["S"] * 2 + ["H"] * 3,
        ["S"] * 10,
    ]
)

# Create DataFrames for hard and soft hand strategies
HARD = pd.DataFrame(
    DH,
    index=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    columns=(2, 3, 4, 5, 6, 7, 8, 9, 10, "A"),
)
SOFT = pd.DataFrame(
    DS, index=[13, 14, 15, 16, 17, 18, 19], columns=(2, 3, 4, 5, 6, 7, 8, 9, 10, "A")
)

# Define a score table for card values
SCORE_TABLE = {
    "A": 11,
    "J": 10,
    "K": 10,
    "Q": 10,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
}

# Define a dictionary for move explanations
EX = {
    "H": "Hit",
    "S": "Stand",
    "Dh": "Double if allowed, otherwise hit",
    "Ds": "Double if allowed, otherwise stand",
    "Rh": "Surrender if allowed, otherwise hit",
}


# Define a class for representing a hand in blackjack
class Hand:
    def __init__(
        self, cards: list, score: int = None, state: str = None, dealer=False
    ) -> None:  # cards is list of cards [2,10] ['J','K']
        """
        Initialize a blackjack hand.

        Args:
            cards (list): List of cards in the hand.
            score (int): Initial score of the hand (optional).
            state (str): State of the hand ('soft' or 'hard') (optional).
            dealer (bool): True if the hand represents the dealer's hand (optional).
        """
        self.state = state
        self.cards = cards
        self.score = score
        if self.cards == ["A", "A"]:
            self.score = 12

        else:
            self.score = sum([SCORE_TABLE[i] for i in self.cards])

        if "A" in self.cards:
            self.state = "soft"
        else:
            self.state = "hard"

    def is_blackjack(self):
        """
        Check if the hand is a blackjack (score equals 21).

        Returns:
            bool: True if it's a blackjack, False otherwise.
        """
        if self.score == 21:
            return True

    def recommend(self, dealer):  # dealer is dealer hand 'J', 'K', 10, 8,...
        """
        Get a recommendation for the next move based on the hand and the dealer's face-up card.

        Args:
            dealer (Hand): The dealer's hand.

        Returns:
            str: Recommended move ('H', 'S', 'Dh', 'Ds', or 'Rh').
        """
        if self.state == "hard":
            table = HARD
            if self.score <= 8:
                score = 8
            elif self.score >= 17:
                score = 17
            else:
                score = self.score
        else:
            table = SOFT
            if self.score <= 13:
                score = 13
            elif self.score >= 19:
                score = 19
            else:
                score = self.score

        response = table.loc[score, dealer.get_score()]
        print(score)
        return response

    def get_score(self):
        """
        Get the current score of the hand.

        Returns:
            int: The hand's score.
        """
        return self.score


def check_winner(player_hand, dealer_hand):
    """
    Check the winner of the blackjack game.

    Args:
        player_hand (Hand): The player's hand.
        dealer_hand (Hand): The dealer's hand.

    Returns:
        str: Result of the game ('Dealer is busted...', 'Dealer and player have blackjack.', etc.).
    """
    if dealer_hand.get_score() > 21:
        return "Dealer is busted. Player wins."
    elif player_hand.is_blackjack() and dealer_hand.is_blackjack():
        return "Dealer and player have blackjack."

    elif dealer_hand.is_blackjack():
        return "Dealer has blackjack. Dealer wins."

    elif (
        dealer_hand.get_score() > 16
        and player_hand.get_score() > dealer_hand.get_score()
        and player_hand.get_score() <= 21
    ):
        return "Player wins."
    elif (
        player_hand.get_score() < dealer_hand.get_score()
        and player_hand.get_score() <= 21
    ):
        return "Dealer wins."
    elif player_hand.get_score() == dealer_hand.get_score():
        return "Tie."
    else:
        return "Dealer keeps playing."
