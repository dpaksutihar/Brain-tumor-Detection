import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K


# Image size used during training
IMG_HEIGHT, IMG_WIDTH = 256, 256 

@register_keras_serializable()
def binary_crossentropy(y_true, y_pred, pos_weight=15):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())  # Avoid log(0)
    
    # Calculate the weighted binary cross-entropy
    loss = - (pos_weight * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

@register_keras_serializable()
def iou_metric(y_true, y_pred, pos_weight=1):
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    # Increase weight for positive class (1s)
    weighted_iou = intersection / (union + tf.keras.backend.epsilon()) * pos_weight
    return weighted_iou

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, pos_weight=1):
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2 * intersection + tf.keras.backend.epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    
    # Increase weight for positive class (1s)
    weighted_dice = dice * pos_weight
    return weighted_dice

def preprocess_image(image_path):
    """Preprocess an image for model prediction."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0  # Normalize to [0,1]
    return np.expand_dims(image, axis=0)

def postprocess_mask(mask):
    """Convert model output to a binary mask."""
    mask = mask.squeeze()
    mask = (mask >= 0.8).astype(np.uint8)  # Thresholding
    return np.array(mask*255, dtype=np.uint8)  # Convert to 0-255 for visualization

# Load trained U-Net model
model = load_model("best_unet_segmentation_model.keras",custom_objects={"iou_metric": iou_metric,
                                                           "dice_coefficient":dice_coefficient,
                                                           'binary_crossentropy':binary_crossentropy})

def predict_and_display(image_path, orignal_mask = None, model = model):
    """Run inference and display results."""
    image = preprocess_image(image_path)
    mask_pred = model.predict(image)[0]
    mask = postprocess_mask(mask_pred)
    print(np.unique(mask,return_counts=True))

    # Display original image and predicted mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
    plt.title("Predicted Mask")
    plt.axis("off")

    if orignal_mask:
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.imread(orignal_mask,cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=255)
        plt.title("Orignal Mask")
        plt.axis("off")

    plt.show()

# Example usage
image_path = "segmentation_data/images/glioma/brainTumorDataPublic_2299-3064_3047.png"
predict_and_display(image_path,
                    orignal_mask = 'segmentation_data/masks/glioma/brainTumorDataPublic_2299-3064_3047.png')
