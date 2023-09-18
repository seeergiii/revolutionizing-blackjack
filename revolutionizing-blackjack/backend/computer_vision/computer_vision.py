import os
import pandas as pd
from roboflow import Roboflow

# Card class mapping
class_ids = [
    "10c",
    "10d",
    "10h",
    "10s",
    "2c",
    "2d",
    "2h",
    "2s",
    "3c",
    "3d",
    "3h",
    "3s",
    "4c",
    "4d",
    "4h",
    "4s",
    "5c",
    "5d",
    "5h",
    "5s",
    "6c",
    "6d",
    "6h",
    "6s",
    "7c",
    "7d",
    "7h",
    "7s",
    "8c",
    "8d",
    "8h",
    "8s",
    "9c",
    "9d",
    "9h",
    "9s",
    "Ac",
    "Ad",
    "Ah",
    "As",
    "Jc",
    "Jd",
    "Jh",
    "Js",
    "Kc",
    "Kd",
    "Kh",
    "Ks",
    "Qc",
    "Qd",
    "Qh",
    "Qs",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

##### OWN YOLO v8 MODEL (0.891% accuracy) ####


##### ROBOFLOW MODEL ####
def load_roboflow_model() -> Roboflow:
    """
    Load and return roboflow model
    """
    api_key = os.environ["ROBOFLOW_API_KEY"]
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
        os.path.join("backend", "computer_vision", "temp_image", img),
        confidence=int(40),
        overlap=int(30),
    ).json()["predictions"]

    card_predictions = (
        pd.DataFrame(card_predictions) if not card_predictions == [] else None
    )
    print("✅ Predictions obtained succesfully")
    print(card_predictions)

    return card_predictions
