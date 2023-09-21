import os

# Parent folder for all project data (model + weight data)
DATA_PATH = os.environ.get("DATA_PATH")

# Folder for train, valid, test data
DATA_DATA_PATH = os.path.join(DATA_PATH, "data")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

# Define paths to model training data
TRAIN_DIR_IMGS = os.path.join(DATA_PATH, "data", "train", "images")
TRAIN_DIR_LABELS = os.path.join(DATA_PATH, "data", "train", "labels")

VALID_DIR_IMGS = os.path.join(DATA_PATH, "data", "valid", "images")
VALID_DIR_LABELS = os.path.join(DATA_PATH, "data", "valid", "labels")

TEST_DIR_IMGS = os.path.join(DATA_PATH, "data", "test", "images")
TEST_DIR_LABELS = os.path.join(DATA_PATH, "data", "test", "labels")

# Define paths to model weight save destinations
WEIGHTS_CHECKPOINTS_PATH = os.path.join(DATA_PATH, "weights_checkpoints")
WEIGHTS_FINAL_PATH = os.path.join(DATA_PATH, "weights_final")
WEIGHTS_START_PATH = os.path.join(DATA_PATH, "weights_start")

# Model params
SPLIT_RATIO = 0.2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
IMAGE_SIZE = 416
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
