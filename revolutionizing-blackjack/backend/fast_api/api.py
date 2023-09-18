from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import os

from backend.computer_vision.computer_vision import (
    load_roboflow_model,
    predict_roboflow_model,
    load_model,
    image_decoder,
    get_predictions,
    class_mapping,
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

# Cache models
app.state.roboflow_model = load_roboflow_model()
app.state.model = load_model(os.environ["MODEL_PATH"])


@app.get("/")
def index():
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
    directory = os.path.join("backend", "computer_vision", "temp_image")
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
