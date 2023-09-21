import os
import pandas as pd
import numpy as np

import keras_cv
import tensorflow as tf

from params import *
from roboflow import Roboflow


##### Load RoboFlow model ####
def load_roboflow_model() -> Roboflow:
    """
    Load and return roboflow model
    """
    api_key = ROBOFLOW_API_KEY
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("playing-cards-ow27d")
    model = project.version(int(4)).model

    print("✅ RoboFlow model loaded succefully")

    return model


def predict_roboflow_model(model: Roboflow, img: str = "input.png") -> pd.DataFrame:
    """
    Returns preedictions based on an input Roboflow model and image.
    """
    print("⏳ Obtaining predictions... Please wait...")
    card_predictions = model.predict(
        os.path.join("computer_vision", "temp_image", img),
        confidence=int(40),
        overlap=int(30),
    ).json()["predictions"]

    card_predictions = (
        pd.DataFrame(card_predictions) if not card_predictions == [] else None
    )
    print("✅ Predictions obtained succesfully")
    print(card_predictions)

    return card_predictions


##### Load own YOLOv8 model ####
def create_custom_model() -> keras_cv.models:
    """
    Builds and returns a YOLOv8 XL Backbone CoCo model structure.
    """
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(CLASS_MAPPING),
        bounding_box_format="center_xywh",
        backbone=backbone,
        fpn_depth=1,
    )

    print("✅ YOLOv8 structure built succesfully")

    return yolo


def load_model(model_weights: str) -> keras_cv.models:
    """
    Loads the weights from a given trained model into the YOLOv8 XL Backbone CoCo model structure.
    """
    reconstructed_yolo = create_custom_model()
    reconstructed_yolo.load_weights(model_weights)

    print("✅ Own YOLOv8 model loaded succesfully")
    return reconstructed_yolo


def load_image(input_img: str) -> tf.Tensor:
    """
    Loads an image from a given path. Returns the image decoded.
    """
    image = tf.io.read_file(input_img)
    decoded_img = tf.image.decode_jpeg(image, channels=3)

    print("✅ Frame decoded succesfully")
    return decoded_img


def image_decoder(input_img: bytes) -> tf.Tensor:
    """
    Decodes an image.
    """
    decoded_img = tf.image.decode_jpeg(input_img, channels=3)

    print("✅ Frame decoded succesfully")
    return decoded_img


def get_predictions(model: keras_cv.models, decoded_img: tf.Tensor) -> tf.Tensor:
    """
    Reshapes and padds a given input image to improve prediction's performance of the model.
    """
    # Get shape of original image
    original_shape = tf.shape(decoded_img)
    original_height = original_shape[0]
    original_width = original_shape[1]

    # Define target size
    target_height = 416
    target_width = 416

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate new dimensions while keeping the aspect ratio
    if original_width > original_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize image, keeping aspect ratio
    resized_img = tf.image.resize(decoded_img, (new_height, new_width))

    # Calculate padding
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    # Pad the resized image with white (you may need to adjust the constant value based on your image)
    constant_values = (
        255 if tf.reduce_max(resized_img) > 1.0 else 1.0
    )  # Adjust based on the max value in your image
    padded_img = tf.pad(
        resized_img,
        paddings=[
            [pad_height // 2, pad_height - pad_height // 2],
            [pad_width // 2, pad_width - pad_width // 2],
            [0, 0],
        ],
        constant_values=constant_values,
    )

    # Expand dimensions to match the model's input shape and cast to float32
    expanded_img = tf.expand_dims(padded_img, axis=0)
    casted_img = tf.cast(expanded_img, tf.float32)

    # Gets predictions from model
    y_pred = model.predict(casted_img)

    if y_pred is None:
        num_detections = 0
        predicted_cards = []
        original_img_boxes = []
        confidences = []
    else:
        num_detections = int(y_pred["num_detections"][0])
        boxes = y_pred["boxes"][0][:num_detections]
        boxes = boxes.astype(int)

        classes = y_pred["classes"][0][:num_detections]
        classes = classes.tolist()

        confidences = y_pred["confidence"][0][:num_detections]
        confidences = confidences.tolist()

        predicted_cards = [CLASS_MAPPING[card] for card in classes]

        # Calculate new dimensions while keeping the aspect ratio
        original_img_w = decoded_img.shape[1]
        original_img_h = decoded_img.shape[0]

        original_img_boxes = []

        for box in boxes:
            if pad_height:
                # Height has been padded
                box_x = box[0] / new_width * original_img_w
                box_w = box[2] / new_width * original_img_w

                box_y = (box[1] - pad_height / 2) / new_height * original_img_h
                box_h = box[3] / new_height * original_img_h

                top_left_x = box_x - box_w
                top_left_y = box_y - box_h

                original_img_boxes.append([top_left_x, top_left_y, box_w, box_h])

            else:
                # Width has been padded
                box_x = (box[0] - pad_width / 2) / new_width * original_img_w
                box_w = box[2] / new_width * original_img_w

                box_y = box[1] / new_height * original_img_h
                box_h = box[3] / new_height * original_img_h

                top_left_x = box_x - box_w
                top_left_y = box_y - box_h

                original_img_boxes.append(
                    [int(top_left_x), int(top_left_y), int(box_w), int(box_h)]
                )

    predictions = {
        "detections": num_detections,
        "cards detected": predicted_cards,
        "bounding boxes": original_img_boxes,
        "confidence": confidences,
    }

    print("✅ Predictions obtained succesfully")
    print(predictions)

    return predictions
