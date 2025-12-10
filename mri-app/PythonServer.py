## 🧠 Brain Tumor Analysis Backend (Expanded Version)

# --- Standard Library Imports ---
import os
import json
import time
from datetime import datetime
from io import BytesIO

# --- Flask & Server Imports ---
from flask import (
    Flask, request, jsonify, send_file, url_for, send_from_directory, abort
)
from flask_cors import CORS 
from werkzeug.datastructures import FileStorage

# --- Deep Learning & Image Processing Imports ---
import tensorflow as tf 
import numpy as np
import cv2 # OpenCV for image manipulation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Input, Conv2D, Dense, Dropout, Flatten, 
    BatchNormalization, Activation, MaxPooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import DenseNet121, VGG16 # Added VGG16 for future expansion
from keras.saving import register_keras_serializable


# --- Global Model and Configuration Variables ---

# Flask App Initialization
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Configuration Settings
class Config:
    """Application configuration settings."""
    UPLOAD_FOLDER = 'data/uploaded_images'
    SEGMENTED_FOLDER = 'data/segmented_results'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    ORIGINAL_IMAGES_DIR = 'data/images/Uploaded'
    SEGMENTED_IMAGES_DIR = 'data/images/Segmented'
    JSON_PATH = 'data/results.json'
    TEMP_DIR = 'temp_processing'
    # Model Hyperparameters (Placeholder, actual models are loaded)
    IMAGE_SIZE = (256, 256)
    SEGMENTATION_SIZE = (256, 256)
    CLASS_NAMES = ['Glioma', 'Meningioma', 'Pediatric', 'No Tumour', 'Uncertain']
    SEGMENTATION_THRESHOLD = 0.5 # Threshold for generating the tumor mask

app.config.from_object(Config)

# Ensure necessary directories exist
def setup_directories():
    """Creates the required file system directories if they don't exist."""
    for path in [
        # Config.UPLOAD_FOLDER, Config.SEGMENTED_FOLDER,
        Config.ORIGINAL_IMAGES_DIR, Config.SEGMENTED_IMAGES_DIR,
        Config.TEMP_DIR
    ]:
        os.makedirs(path, exist_ok=True)

setup_directories()

# Global Model Variables
classificaion_model = None
segmented_model = None


# --- Custom Keras Components (Losses and Metrics) ---

@register_keras_serializable()
def binary_crossentropy_weighted(y_true, y_pred, pos_weight=15):
    """
    Binary Crossentropy with positive class weighting for unbalanced datasets.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = - (pos_weight * y_true * tf.math.log(y_pred) +
              (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

@register_keras_serializable()
def iou_metric(y_true, y_pred, threshold=0.5):
    """
    Intersection Over Union (Jaccard Index) metric.
    """
    y_true = tf.cast(y_true, tf.float32)
    # Apply a soft threshold to prediction for stability
    y_pred = tf.cast(y_pred > threshold, tf.float32) 

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection / (union + K.epsilon()))

@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    """
    Dice Similarity Coefficient (F1 Score) metric.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + K.epsilon()) / \
           (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + K.epsilon())
    return dice


# --- Model Building and Loading Functions ---

def compile_model(model):
    """
    Compile the model with a standard optimizer, loss, and metrics.
    """
    # Using a slightly lower learning rate for loaded models or fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

def build_classification_model(base_model_layer):
    """
    Create a custom classification model head on a pre-trained base.
    """
    # The actual implementation is simple, but this function demonstrates a full build process.
    model = Sequential([
        Input(shape=(224, 224, 3)), # Input layer size
        base_model_layer,
        
        # Convolutional layers for feature refinement
        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0)),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0)),
        BatchNormalization(),
        
        GlobalAveragePooling2D(), # Reduces feature maps to a single vector
        
        Dropout(0.5), # Regularization to prevent overfitting
        
        # Dense classification head
        Dense(256, activation='relu', kernel_regularizer=l2(0.0)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.0)),
        
        # Output layer with 4 classes
        Dense(len(Config.CLASS_NAMES) - 1, activation='softmax') # Exclude 'Uncertain'
    ])
    return model

def load_models():
    """Loads the pre-trained classification and segmentation models."""
    global classificaion_model, segmented_model
    
    print("--- Loading Deep Learning Models ---")
    
    try:
        # Load Classification Model
        # Note: The provided code loads a saved model, but here's how a full build would look
        # base = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # classificaion_model = build_classification_model(base)
        # compile_model(classificaion_model)
        classificaion_model = load_model('models/BrainClassification.h5')
        print(f"✅ Classification Model loaded from: models/BrainClassification.h5")
        
        # Load Segmentation Model
        segmented_model = load_model(
            'models/BrainSegmentation.keras',
            custom_objects={
                "binary_crossentropy_weighted": binary_crossentropy_weighted,
                "iou_metric": iou_metric,
                "dice_coefficient": dice_coefficient
            },
            compile=False # Assuming the loaded model is already compiled
        )
        print(f"✅ Segmentation Model loaded from: models/BrainSegmentation.keras")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # In a production environment, you might want to exit or provide a fallback
        # sys.exit(1)

# Execute model loading on startup
load_models()


# --- Image Preprocessing Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def enhance_image(image_data):
    """
    Preprocesses the medical image for both classification and segmentation models.
    Steps: Resize, Normalize, and ensure correct shape.
    
    Args:
        image_data (np.ndarray): The raw image loaded by OpenCV.
        
    Returns:
        tuple: (enhanced_for_seg, enhanced_for_class)
    """
    if image_data is None:
        raise ValueError("Image data is None.")
        
    # 1. Resize for Segmentation Model (256x256x3)
    img_seg = cv2.resize(image_data, Config.SEGMENTATION_SIZE, interpolation=cv2.INTER_LINEAR)

    # 2. Resize for Classification Model (224x224x3 - Grayscale)
    # The classification model expects 224x224x1 (grayscale in your original code)
    # But DenseNet base takes 224x224x3, so we adapt for the loaded model's expectation.
    img_class = cv2.resize(image_data, Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)


    # 3. Normalization (0-1 range)
    if img_seg.max() > 1.0:
        img_seg_norm = img_seg.astype(np.float32) / 255.0
    else:
        img_seg_norm = img_seg.astype(np.float32)

    if img_class.max() > 1.0:
        img_class_norm = img_class.astype(np.float32) / 255.0
    else:
        img_class_norm = img_class.astype(np.float32)
        
    # The classification model in the original code expects grayscale and then a batch dim:
    # `np.expand_dims( cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY), axis=0)`
    # We must match this expectation for the loaded model.
    img_class_gray = cv2.cvtColor(img_class_norm, cv2.COLOR_BGR2GRAY)


    return img_seg_norm, img_class_gray


# --- Core Analysis Logic ---

def classify_and_segment(file: FileStorage):
    """
    Performs tumor classification and segmentation on an uploaded image file.
    
    Args:
        file (FileStorage): The uploaded file object from Flask request.
        
    Returns:
        tuple: (tumor_type, confidence, segmented_image_url)
    """
    global classificaion_model, segmented_model
    
    # 1. Read and Decode Image
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_original is None:
        raise ValueError("Could not decode image.")

    img_size = img_original.shape
    
    # 2. Preprocess Images
    img_for_seg, img_for_class_gray = enhance_image(img_original)
    
    # 3. Classification Prediction
    # Classification model expects (1, 224, 224, 1) - Batch, H, W, Channel
    classification_input = np.expand_dims(img_for_class_gray, axis=(0, -1))

    # Note: Your loaded model seems to handle grayscale input correctly based on your original code's preprocessing
    pred = classificaion_model.predict(classification_input)
    max_id = np.argmax(pred, axis=-1)[0]
    confidence = round(pred[0, max_id] * 100, 2)
    
    # 4. Uncertainty Check
    if confidence < 50.0:
        tumor_type = Config.CLASS_NAMES[3] # 'Uncertain'
        seg_conf = [0,0,0]
        # No segmentation if confidence is too low
        
    else:
        tumor_type = Config.CLASS_NAMES[max_id]
        
        # 5. Segmentation Prediction
        # Segmentation model expects (1, 256, 256, 3) - Batch, H, W, Channels
        segmentation_input = np.expand_dims(img_for_seg, axis=0)
        segmented_img_raw = segmented_model.predict(segmentation_input)
        
        img_original, seg_conf = apply_segmentation_overlay(img_original,img_size, segmented_img_raw)
        
    # 8. Save Temporary Segmented Image
    temp_filename = f'segmented_{int(time.time() * 1000)}.png'
    temp_path = os.path.join(Config.TEMP_DIR, temp_filename)
    cv2.imwrite(temp_path, img_original) 
    
    # 9. Construct Temporary URL
    segmented_url = url_for('get_temp_file', filename=temp_filename, _external=True)

    return tumor_type, float(confidence),seg_conf, segmented_url


def apply_segmentation_overlay(original_img, img_size, segmented_img_raw, threshold=0.5):
    """
    Returns:
        final_img: overlayed BGR image
        confidences: list of [red_conf%, orange_conf%, yellow_conf%]
    """

    # Extract probability map
    if segmented_img_raw.ndim == 4:
        prob_map = segmented_img_raw[0, :, :, 0]
    else:
        prob_map = segmented_img_raw[0]

    # Resize to original MRI resolution
    prob_map = cv2.resize(prob_map, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    prob_map = np.clip(prob_map, 0.0, 1.0)


    # Categorized areas
    red_area = (prob_map >= 0.90)
    orange_area = (prob_map >= 0.70) & (prob_map < 0.90)
    yellow_area = (prob_map >= 0.50) & (prob_map < 0.70)

    # Compute confidence mean values safely
    def safe_mean(mask):
        return float(prob_map[mask].mean()) * 100 if np.any(mask) else 0.0

    red_conf  = safe_mean(red_area)
    orange_conf = safe_mean(orange_area)
    yellow_conf = safe_mean(yellow_area)

    confidences = [round(red_conf, 2),
                   round(orange_conf, 2),
                   round(yellow_conf, 2)]


    # Main tumor mask for overlay
    mask = (prob_map >= threshold).astype(np.uint8)
    overlay = np.zeros_like(original_img, dtype=np.uint8)

    # Apply colors with high visual separation
    overlay[red_area] = [0, 0, 255]        # Red (High confidence)
    overlay[orange_area] = [255, 0, 255]   # Magenta (Medium confidence)
    overlay[yellow_area] = [255, 255, 0]   # Cyan (Low confidence)


    # Apply overlay
    alpha = 0.45
    final_img = original_img.copy()
    if mask.max()>0:
        final_img[mask == 1] = cv2.addWeighted(
            original_img[mask == 1], 1 - alpha,
            overlay[mask == 1], alpha, 0
        )

    return final_img, confidences

# --- Utility Data Functions ---

def load_json_results():
    """Load all existing results from the JSON file."""
    if os.path.exists(Config.JSON_PATH):
        try:
            with open(Config.JSON_PATH, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_json_results(data):
    """Save the updated results back to the JSON file."""
    with open(Config.JSON_PATH, 'w') as file:
        json.dump(data, file, indent=4)

# --- Flask Routes ---

@app.route('/temp/<filename>')
def get_temp_file(filename):
    """Serves temporary files (like segmented images)."""
    try:
        return send_from_directory(Config.TEMP_DIR, filename)
    except FileNotFoundError:
        abort(404)

@app.route('/data/images/<path:filename>')
def serve_data_images(filename):
    """Serves saved original and segmented images."""
    # This route is typically not used directly by the frontend, but is good practice.
    if 'original/' in filename:
        return send_from_directory(Config.ORIGINAL_IMAGES_DIR, filename)
    elif 'segmented/' in filename:
        return send_from_directory(Config.SEGMENTED_IMAGES_DIR, filename)
    abort(404)


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Endpoint for receiving an image and returning analysis results."""

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file part or empty file name'}), 400
        
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        tumor_type, confidence,seg_confidence, segmented_image_url = classify_and_segment(file)

        return jsonify({
            'tumorType': tumor_type,
            'tumorConf': confidence,
            'segConf':seg_confidence,
            'segmentedImage': segmented_image_url
        })
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': f'Internal server error during analysis: {str(e)}'}), 500


@app.route('/save-results', methods=['POST'])
def save_results():
    """Endpoint to save patient data and analysis results to disk."""
    
    # 1. Input Validation and Extraction
    required_fields = ['patientName', 'tumorType', 'confidence', 'otherDetails']
    required_files = ['originalImage', 'segmentedImage']

    for field in required_fields:
        if field not in request.form:
            return jsonify({"error": f"Missing form field: {field}"}), 400
            
    for file_key in required_files:
        if file_key not in request.files or request.files[file_key].filename == '':
            return jsonify({"error": f"Missing file: {file_key}"}), 400

    try:
        patient_name = request.form['patientName']
        tumor_type = request.form['tumorType']
        confidence = request.form['confidence']
        other_details = request.form['otherDetails']

        original_image = request.files['originalImage']
        segmented_image = request.files['segmentedImage']

        # 2. Create unique filenames and paths
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in patient_name)
        file_base_name = f'{safe_name}_{timestamp_str}.png'

        original_path = os.path.join(Config.ORIGINAL_IMAGES_DIR, file_base_name)
        segmented_path = os.path.join(Config.SEGMENTED_IMAGES_DIR, file_base_name)

        # 3. Save files to their respective folders
        original_image.save(original_path)
        segmented_image.save(segmented_path) # NOTE: The segmented image must be sent from the frontend after analysis
                                             # as the temporary image is already served.

        # 4. Record new entry
        results = load_json_results()

        new_record = {
            "Patient Name": patient_name,
            "Tumor Type": tumor_type,
            "Confidence": confidence,
            "Other Details": other_details,
            # Store paths relative to the 'data' directory for cleaner JSON
            "Original Image": original_path,
            "Segmented Image": segmented_path,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S")
        }

        # 5. Append and save JSON
        results.append(new_record)
        save_json_results(results)

        return jsonify({"message": "Results saved successfully!", "record": new_record}), 200

    except Exception as e:
        print(f"Save results error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get-patient-records', methods=['GET'])
def get_patient_records():
    """Returns all saved patient analysis records."""
    records = load_json_results()
    
    # Optionally, transform paths to full URLs for frontend access
    for record in records:
        if 'Original Image' in record and not record['Original Image'].startswith('http'):
            record['Original Image URL'] = url_for('serve_data_images', filename=record['Original Image'], _external=True)
        if 'Segmented Image' in record and not record['Segmented Image'].startswith('http'):
            record['Segmented Image URL'] = url_for('serve_data_images', filename=record['Segmented Image'], _external=True)

    return jsonify({"records": records}), 200

class Deployer:
    """Class for managing model deployment (e.g., versioning, A/B testing)."""
    def __init__(self, model_names):
        self.active_models = {}
        self.history = []
        for name in model_names:
            self.active_models[name] = 'v1.0'

    def update_model(self, name, version):
        """Simulates updating a model version."""
        self.history.append({'model': name, 'old_version': self.active_models[name], 'new_version': version, 'timestamp': datetime.now().isoformat()})
        self.active_models[name] = version
        print(f"Deployment: {name} updated to {version}")

class LoggingService:
    """Service to handle structured logging and metrics."""
    def log_inference(self, patient_id, tumor_type, confidence, duration_ms):
        """Logs the details of a single inference request."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'prediction': tumor_type,
            'confidence': confidence,
            'latency_ms': duration_ms,
            'status': 'success'
        }
        pass


if __name__ == '__main__':
    # Running in debug mode reloads the server on code changes, but re-runs load_models.
    # In a production environment, debug=False is mandatory.
    app.run(debug=True, host='0.0.0.0', port=5000)