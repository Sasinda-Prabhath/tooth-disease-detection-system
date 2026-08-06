import keras_cv
import tensorflow as tf


def build_molar_detector(num_classes: int = 32):
    """Full YOLOv8 detector — use when COCO tooth dataset is ready."""
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
    model = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=2,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        classification_loss="binary_crossentropy",
        box_loss="ciou",
    )
    return model
