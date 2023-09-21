import keras_cv
import tensorflow as tf
from tensorflow import keras
import time

from computer-vision-training.params import *
from computer-vision-training.cloud import save_model_to_gcloud


def build_compile_model() -> keras_cv.models.YOLOV8Detector:
    """
    Build and compile a YOLOv8 model for object detection.

    Returns:
        keras_cv.models.YOLOV8Detector: The compiled YOLOv8 model.
    """

    # yolov8 backbone with coco weights
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")

    # Build the model
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(CLASS_MAPPING),
        bounding_box_format="center_xywh",
        backbone=backbone,
        fpn_depth=1,  # Feature Pyramid Network
    )

    # Compile the model with custom adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )

    yolo.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )

    print("✅ Model built successfully")

    return yolo


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """
    A custom Keras callback for evaluating and saving model checkpoints based on COCO metrics.

    Args:
        data (tf.data.Dataset): The dataset for evaluation.
        save_path (str): The directory where model checkpoints will be saved.
    """

    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        """
        Called at the end of each epoch for evaluation and checkpoint saving.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing training logs.
        """
        self.metrics.reset_state()
        for batch in self.data:
            images = batch["images"]
            bounding_boxes = batch["bounding_boxes"]

            # Extract "boxes" and "classes" from bounding_boxes
            classes = bounding_boxes["classes"]
            boxes = bounding_boxes["boxes"]

            y_pred = self.model.predict(images, verbose=0)

            # Convert classes and bounding_boxes to a dictionary
            y_true = {"classes": classes, "boxes": boxes}

            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics[
            "MaP@[area=small]"
        ]  # set target metric as MAP on small boxes

        # If the current model as better MAP on small boxes
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)
            timestamp = time.strftime("%Y%m%d-%H%M")

            # Create model path
            model_path = os.path.join(
                self.save_path,
                f"map_small_{current_map}_model_weights_checkpoint.h5",
            )

            # Save model locally and to GCloud
            self.model.save_weights(model_path)
            print("✅ Model checkpoint saved successfully locally")
            save_model_to_gcloud(model_path, BUCKET_NAME)

        return logs
