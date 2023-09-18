import streamlit as st
from streamlit import session_state
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import av
import cv2
import os
import numpy as np
import requests
from roboflow import Roboflow


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def get_coordinates_of_clusters(
    frame, canny_thresh1: float = 50, canny_thresh2: float = 150
):
    image_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_greyscale, (5, 5), 0)
    image_canny_kernel = cv2.Canny(image_blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(
        image_canny_kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    preprocessed_image_size = image_greyscale.shape[0] * image_greyscale.shape[1]

    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio_w_h = w / h
        aspect_ratio_h_w = h / w

        # Change these thresholds as per your requirements
        area_threshold = 0.02 * preprocessed_image_size
        aspect_ratio_threshold = 0.4

        if (
            area > area_threshold
            and aspect_ratio_w_h > aspect_ratio_threshold
            and aspect_ratio_h_w > aspect_ratio_threshold
        ):
            coordinates.append((x, y, w, h))

    return coordinates


class VideoProcessor:
    def __init__(self):
        self.player_midpoint = None  # Initialize the player's midpoint
        self.dealer_midpoint = None  # Initialize the dealer's midpoint
        self.frame = None  # Initialize a frame

        self.canny_thresh1 = 50
        self.canny_thresh2 = 150
        self.canny_image_stream = False
        self.alpha = 0.5
        self.beta = 0

        self.frame_counter = 0  # Initialize the frame counter

    def recv(self, frame):
        self.frame_counter += 1
        img_for_save = frame.to_ndarray(format="bgr24")
        img = np.copy(img_for_save)

        if self.canny_image_stream:
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


st.title("Revolutionizing Blackjack :heart:")
st.write("The ultimate blackjack AI")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

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

st.sidebar.title("Press predict when you are ready! :sunglasses:")
st.sidebar.title("⬇️⬇️⬇️")
button = st.sidebar.button("Predict")


if button:
    # save queued values
    frame_at_button_press = webrtc_ctx.video_transformer.frame
    player_midpoint_at_button_press = webrtc_ctx.video_transformer.player_midpoint
    dealer_midpoint_at_button_press = webrtc_ctx.video_transformer.dealer_midpoint

    # Check if frame_at_button_press is valid (not None and not empty)
    if frame_at_button_press is not None and frame_at_button_press.size > 0:
        # Temporary saving frame to make API call
        img_directory = os.path.join("frontend", "temp_image")
        img_name = "frame.png"
        img_path = os.path.join(img_directory, img_name)
        cv2.imwrite(img_path, frame_at_button_press)

        # Create a dictionary with the image file
        files = {"img": ("frame.png", open(img_path, "rb"))}
        # Make the API call
        api_url = os.environ["ENDPOINT_RB"]
        response = requests.post(api_url, files=files)

        # Check the response
        if response.status_code == 200:
            predictions = response.json()
            print("✅ API called succesfully")
            print(predictions)
        else:
            print("❌ API call failed with status code:", response.status_code)

        os.remove(img_path)

        player_cards = []
        dealer_cards = []

        print(predictions["detections"])

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
            print(player_cards)
            print(dealer_cards)

            st.sidebar.title("Player cards:")

            player_cards_string = ""

            for card in player_cards:
                emoji = (
                    "♣️"
                    if card[-1] == "C"
                    else "♠️"
                    if card[-1] == "S"
                    else "♥️"
                    if card[-1] == "H"
                    else "♦️"
                    if card[-1] == "D"
                    else ""
                )
                player_cards_string += " " + card[:-1] + emoji

            st.sidebar.title(player_cards_string)

            st.sidebar.title("Dealer cards:")

            dealer_cards_string = ""

            for card in dealer_cards:
                emoji = (
                    "♣️"
                    if card[-1] == "C"
                    else "♠️"
                    if card[-1] == "S"
                    else "♥️"
                    if card[-1] == "H"
                    else "♦️"
                    if card[-1] == "D"
                    else ""
                )
                dealer_cards_string += " " + card[:-1] + emoji

            st.sidebar.title(dealer_cards_string)

            headers = {"accept": "application/json", "Content-Type": "application/json"}

            data = {"dealer": dealer_cards, "player": player_cards}

            response = requests.post(
                "https://recommend-okumlrfyiq-ew.a.run.app/predict_move",
                headers=headers,
                json=data,
            )

            st.sidebar.title("Recommended move:")

            try:
                if not response.json()["next_move"] == "":
                    st.sidebar.title(response.json()["next_move"])
                else:
                    st.sidebar.title("Game ended, youre busted!")
            except:
                st.sidebar.title("Game ended, youre busted!")
