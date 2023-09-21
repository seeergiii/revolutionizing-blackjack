import streamlit as st
from streamlit import session_state
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import av
import cv2
import os
import numpy as np
import requests

# Modify for working API URL
api_url = "http://127.0.0.1:8000"

# RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Function to get coordinates of card clusters in a frame
def get_coordinates_of_clusters(
    frame, canny_thresh1: float = 50, canny_thresh2: float = 150
) -> list:
    """
    Get the coordinates of clusters in an image frame.

    Args:
        frame (numpy.ndarray): Input image frame.
        canny_thresh1 (float): Threshold for Canny edge detection.
        canny_thresh2 (float): Threshold for Canny edge detection.

    Returns:
        list: List of tuples representing the coordinates of clusters.
    """
    # Preprocess the frame for object detection
    image_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, canny_thresh1, canny_thresh2)

    # Find contours of objects in the frame
    contours, _ = cv2.findContours(
        image_canny_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate the size of the preprocessed image
    preprocessed_image_size = image_greyscale.shape[0] * image_greyscale.shape[1]

    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio_w_h = w / h
        aspect_ratio_h_w = h / w

        # Define thresholds for object filtering
        # Change thresholds as per your requirements
        area_threshold = 0.02 * preprocessed_image_size
        aspect_ratio_threshold = 0.4

        # Filter objects based on area and aspect ratio
        if (
            area > area_threshold
            and aspect_ratio_w_h > aspect_ratio_threshold
            and aspect_ratio_h_w > aspect_ratio_threshold
        ):
            coordinates.append((x, y, w, h))

    return coordinates


odds_dealer_cards = []  # List of dealer cards
odds_player_cards = []  # List of player cards
odds_moves = []  # List of moves


# Class to process video frames and perform object detection
class VideoProcessor:
    def __init__(self):
        # Initialize variables for object midpoints and frame
        self.player_midpoint = None  # Initialize the player's midpoint
        self.dealer_midpoint = None  # Initialize the dealer's midpoint
        self.frame = None  # Initialize a frame

        # Default parameters for Canny edge detection
        self.canny_thresh1 = 50
        self.canny_thresh2 = 150
        self.canny_image_stream = False
        self.alpha = 0.5
        self.beta = 0

        # Initialize frame counter
        self.frame_counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Receive and process video frames.

        Args:
            frame (av.VideoFrame): Input video frame.

        Returns:
            av.VideoFrame: Processed video frame.
        """
        self.frame_counter += 1
        img_for_save = frame.to_ndarray(format="bgr24")
        img = np.copy(img_for_save)

        if self.canny_image_stream:
            # Apply Canny edge detection if enabled
            image_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
            img_canny = cv2.Canny(image_blurred, self.canny_thresh1, self.canny_thresh2)
            img = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

        coordinates = get_coordinates_of_clusters(
            img, self.canny_thresh1, self.canny_thresh2
        )

        # Sort coordinates based on area in descending order
        sorted_coordinates = sorted(
            coordinates, key=lambda x: x[2] * x[3], reverse=True
        )

        # Initialize temporary midpoints
        temp_player_midpoint = None
        temp_dealer_midpoint = None

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        padding = 5  # Padding around the text

        if len(sorted_coordinates) >= 2:
            for i, (x, y, w, h) in enumerate(
                sorted_coordinates[:2]
            ):  # Only consider the two largest shapes
                midpoint = (x + w // 2, y + h // 2)

                if i == 0:
                    temp_player_midpoint = midpoint
                elif i == 1:
                    temp_dealer_midpoint = midpoint

        # If both classes are present, initialize the midpoints and the frame
        if temp_player_midpoint and temp_dealer_midpoint:
            self.player_midpoint = temp_player_midpoint
            self.dealer_midpoint = temp_dealer_midpoint
            self.frame = np.copy(img_for_save)

        for i, (x, y, w, h) in enumerate(sorted_coordinates):
            midpoint = (x + w // 2, y + h // 2)

            label = ""
            if self.player_midpoint and self.dealer_midpoint:
                distance_to_player = np.sqrt(
                    (midpoint[0] - self.player_midpoint[0]) ** 2
                    + (midpoint[1] - self.player_midpoint[1]) ** 2
                )
                distance_to_dealer = np.sqrt(
                    (midpoint[0] - self.dealer_midpoint[0]) ** 2
                    + (midpoint[1] - self.dealer_midpoint[1]) ** 2
                )
                label = (
                    "Player Hand"
                    if distance_to_player < distance_to_dealer
                    else "Dealer Hand"
                )

            # Draw rectangles and labels only for large shapes
            if label:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                bg_rect_top_left = (x - padding, y - text_size[1] - 10 - padding)
                bg_rect_bottom_right = (x + text_size[0] + padding, y - 10 + padding)
                cv2.rectangle(
                    img, bg_rect_top_left, bg_rect_bottom_right, (0, 0, 0), -1
                )  # Black background
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )  # White text

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit app title and description
st.title("Revolutionizing Blackjack :heart:")
st.write("The ultimate blackjack AI")

# Create a WebRTC streaming context
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

# Toggle for enabling Canny edge detection
canny_image_stream_global = st.toggle(
    " ⬅️ This switch turns on canny edge detection in videostream factory"
)

# Initialize session state
if "settings_visible" not in session_state:
    session_state.settings_visible = False

# Create a button to toggle the visibility of settings
if st.button("Toggle Settings"):
    session_state.settings_visible = (
        not session_state.settings_visible
    )  # Toggle the visibility

if session_state.settings_visible:
    canny_thresh1 = st.slider(
        "Set Canny Threshold 1 for cluster detection (Default: 50)", 0, 300, value=50
    )
    canny_thresh2 = st.slider(
        "Set Canny Threshold 2 for cluster detection (Default: 150)", 0, 300, value=150
    )

    alpha = st.slider("Set contrast (Default: 50)", 0, 200, value=50)

    beta = st.slider("Set brightness (Default: 0)", -127, 127, value=0)

try:
    if canny_thresh1 is not None:
        webrtc_ctx.video_transformer.canny_thresh1 = canny_thresh1

    if canny_thresh2 is not None:
        webrtc_ctx.video_transformer.canny_thresh2 = canny_thresh2
except:
    pass

try:
    if alpha is not None:
        webrtc_ctx.video_transformer.alpha = alpha / 100
except:
    pass

try:
    if beta is not None:
        webrtc_ctx.video_transformer.beta = beta
except:
    pass


try:
    webrtc_ctx.video_transformer.canny_image_stream = canny_image_stream_global
except:
    pass

st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);
}
</style>""",
    unsafe_allow_html=True,
)

st.sidebar.title("Press predict whenever you are ready! :sunglasses:")
st.sidebar.title("⬇️⬇️⬇️")

button = st.sidebar.button("Predict")
choose_yolo = st.sidebar.toggle(
    "🚧 WORK IN PROGRESS 🚧 - Predict using our own trained YOLOv8 computer vision model"
)

if button:
    # Save queued values
    frame_at_button_press = webrtc_ctx.video_transformer.frame
    player_midpoint_at_button_press = webrtc_ctx.video_transformer.player_midpoint
    dealer_midpoint_at_button_press = webrtc_ctx.video_transformer.dealer_midpoint

    # Check if frame_at_button_press is valid (not None and not empty)
    if frame_at_button_press is not None and frame_at_button_press.size > 0:
        # Temporary saving frame to make API call
        img_directory = os.path.join("temp_image")
        img_name = "frame.png"
        img_path = os.path.join(img_directory, img_name)
        cv2.imwrite(img_path, frame_at_button_press)
        # Create a dictionary with the image file
        files = {"img": ("frame.png", open(img_path, "rb"))}

        # User did not choose to run own YOLO model
        if not choose_yolo:
            # Make the API call
            response = requests.post(
                f"{api_url}/card_predictions_roboflow", files=files
            )

            # Check the response
            if response.status_code == 200:
                predictions = response.json()
                print("✅ API called succesfully")
            else:
                print("❌ API call failed with status code:", response.status_code)

            os.remove(img_path)

        # User chosed using own YOLO model
        if choose_yolo:
            # Make the API call
            api_url = os.environ["ENDPOINT_YOLO"]
            response = requests.post(api_url, files=files)

            # Check the response
            if response.status_code == 200:
                predictions = response.json()
                print("✅ API called succesfully")
            else:
                print("❌ API call failed with status code:", response.status_code)

        player_cards = []
        dealer_cards = []

        if predictions["detections"] > 0:
            for card in range(predictions["detections"]):
                x = predictions["bounding boxes"][card][0]
                y = predictions["bounding boxes"][card][1]
                card_class = predictions["cards detected"][card]

                distance_to_player = np.sqrt(
                    (x - player_midpoint_at_button_press[0]) ** 2
                    + (y - player_midpoint_at_button_press[1]) ** 2
                )

                distance_to_dealer = np.sqrt(
                    (x - dealer_midpoint_at_button_press[0]) ** 2
                    + (y - dealer_midpoint_at_button_press[1]) ** 2
                )

                if distance_to_player >= distance_to_dealer:
                    dealer_cards.append(card_class)
                else:
                    player_cards.append(card_class)

            player_cards = list(set(player_cards))
            dealer_cards = list(set(dealer_cards))

            player_cards_string = ""

            for card in player_cards:
                emoji = (
                    "♣️"
                    if card[-1].upper() == "C"
                    else "♠️"
                    if card[-1].upper() == "S"
                    else "♥️"
                    if card[-1].upper() == "H"
                    else "♦️"
                    if card[-1].upper() == "D"
                    else ""
                )
                player_cards_string += " " + card[:-1] + emoji

            st.sidebar.title(f"Player cards: {player_cards_string}")

            dealer_cards_string = ""

            for card in dealer_cards:
                emoji = (
                    "♣️"
                    if card[-1].upper() == "C"
                    else "♠️"
                    if card[-1].upper() == "S"
                    else "♥️"
                    if card[-1].upper() == "H"
                    else "♦️"
                    if card[-1].upper() == "D"
                    else ""
                )
                dealer_cards_string += " " + card[:-1] + emoji

            st.sidebar.title(f"Dealer cards: {dealer_cards_string}")

            data = {"dealer": dealer_cards, "player": player_cards}

            # Append cards to list to later estimate winning odds
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
            for card in dealer_cards:
                mapped_score = SCORE_TABLE.get(card[:-1])
                odds_dealer_cards.append(int(mapped_score))
            for card in player_cards:
                mapped_score = SCORE_TABLE.get(card[:-1])
                odds_dealer_cards.append(int(mapped_score))

            api_url = os.environ["ENDPOINT_MOVE"]
            response = requests.post(
                os.environ["ENDPOINT_MOVE"],
                json=data,
            )

            if response.status_code == 200:
                moves = response.json()
                print("✅ API called succesfully")
            else:
                print("❌ API call failed with status code:", response.status_code)

            st.sidebar.title(f"Recommended move: {moves['next_move']}")
            st.sidebar.title(moves["message"])

            # Prepare data to call API end-point for XGBoost odds estimator model
            data = {
                "player_card_1": [0],  # Player's first card
                "player_card_2": [0],  # Player's second card
                "player_card_3": [0],  # Player's third card
                "player_card_4": [0],  # Player's fourth card
                "player_card_5": [0],  # Player's fifth card
                "player_card_6": [0],  # Player's sixth card
                "player_card_7": [0],  # Player's seventh card
                "dealer_card_1": [0],  # Dealer's first card
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
                "action_taken_2_None": [0],  # Action taken on second hand
                "action_taken_3_D": [0],  # Action taken on third hand (Dealer)
                "action_taken_3_H": [0],  # Action taken on third hand (Hit)
                "action_taken_3_P": [0],  # Action taken on third hand (Split)
                "action_taken_3_S": [0],  # Action taken on third hand (Stand)
                "action_taken_3_None": [0],  # Action taken on third hand
                "action_taken_4_D": [0],  # Action taken on fourth hand (Dealer)
                "action_taken_4_H": [0],  # Action taken on fourth hand (Hit)
                "action_taken_4_S": [0],  # Action taken on fourth hand (Stand)
                "action_taken_4_None": [0],  # Action taken on fourth hand
                "action_taken_5_H": [0],  # Action taken on fifth hand (Hit)
                "action_taken_5_S": [0],  # Action taken on fifth hand (Stand)
                "action_taken_5_None": [0],  # Action taken on fifth hand
                "action_taken_6_H": [0],  # Action taken on sixth hand (Hit)
                "action_taken_6_S": [0],  # Action taken on sixth hand (Stand)
                "action_taken_6_None": [0],  # Action taken on sixth hand
            }

            # Iterate through dealer cards and update the corresponding keys in the dictionary
            for i, card_value in enumerate(odds_dealer_cards):
                key = f"dealer_card_{i + 1}"
                data[key] = [card_value]

            # Iterate through player cards and update the corresponding keys in the dictionary
            for i, card_value in enumerate(odds_player_cards):
                key = f"player_card_{i + 1}"
                data[key] = [card_value]

            if moves["message"] == "":
                # Map of moves
                EX = {
                    "Hit": "H",
                    "Stand": "S",
                    "Double if allowed, otherwise hit": "D",
                    "Double if allowed, otherwise stand": "D",
                    "Surrender if allowed, otherwise hit": "R",
                }

                # Iterate through moves and update the corresponding keys in the dictionary
                for i, move in enumerate(odds_moves):
                    hand_num = (
                        i // 5
                    ) + 1  # Determine the hand number based on the index
                    mapped_move = None  # Initialize as None in case no match is found
                    for key, value in EX.items():
                        if value == move:
                            mapped_move = key
                            break  # Exit the loop when a match is found
                    move_key = f"action_taken_{hand_num}_{mapped_move}"
                    data[move_key] = [1]

            # Call API end-point for XGBoost odds estimator model
            api_url = os.environ["ENDPOINT_ODDS"]
            response = requests.post(
                os.environ["ENDPOINT_ODDS"],
                json=data,
            )

            if response.status_code == 200:
                odds = response.json()
                print("✅ API called succesfully")
            else:
                print("❌ API call failed with status code:", response.status_code)

            if (
                moves["next_move"] != "Player is busted."
                and moves["next_move"] != "Player got a Blackjack."
            ) or moves["message"] == "Dealer hits.":
                st.sidebar.title(
                    f"🔮 You have a {round(odds['win']*100,2)}% chance of winning this hand"
                )
            else:
                odds_moves = []  # List of moves

        if predictions["detections"] == 0:
            st.sidebar.title("❌ No cards detected!")
            st.sidebar.caption("💡 Did you try improving the lightning?")
            st.sidebar.caption("📹 Are you holding the camera steady enough?")
            st.sidebar.caption("🤔 Are there any other objects on the table?")
            st.sidebar.caption(
                "⚙️ You can always run the predictions on the default model (this is a work-in-progress model!)"
            )
