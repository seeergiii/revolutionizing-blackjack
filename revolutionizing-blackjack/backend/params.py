import os

# LOCAL MODEL PATH
MODEL_PATH = os.environ.get("MODEL_PATH")

# Roboflow
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# Model params
CLASS_IDS = [
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
CLASS_MAPPING = dict(zip(range(len(CLASS_IDS)), CLASS_IDS))
