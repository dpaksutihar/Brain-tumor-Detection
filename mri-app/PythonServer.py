from flask import Flask, request, jsonify, send_file, url_for, send_from_directory
import os
import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
import json
from werkzeug.datastructures import FileStorage

# Build Models
import tensorflow as tf  # For deep learning
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (GlobalAveragePooling2D,
    Input, Conv2D, Dense, Dropout, Flatten)
from tensorflow.keras.regularizers import l2
from datetime import datetime
from tensorflow.keras import backend as K

global classificaion_model, segmented_model
from keras.saving import register_keras_serializable
from flask_cors import CORS 
app = Flask(__name__)
# Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'data/imaged'
app.config['Segmented_FOLDER'] = 'data/segmented'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# Directory paths
ORIGINAL_IMAGES_DIR = 'data/images'
SEGMENTED_IMAGES_DIR = 'data/segmented'
JSON_PATH = 'data/results.json'
TEMP_DIR = 'temp'
CORS(app)  #

# Ensure directories exist
os.makedirs(ORIGINAL_IMAGES_DIR, exist_ok=True)
os.makedirs(SEGMENTED_IMAGES_DIR, exist_ok=True)


# Compile Models
def compile_model(model):
    """
    Compile the model with Adam optimizer and categorical crossentropy loss.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), # Next step in gradient updation
                  loss='categorical_crossentropy', # loss uses in model learning while backpropogation
                  metrics=['accuracy'])  # model evaluation metrics

# def build_model(base_model):
#     """
#     Create a model using a pretrained base model (DenseNet) and add a custom classifier.
#     """
#     model = Sequential([
#     Input(shape=(224,224,3)),
#     base_model,
#     # Dropout(0.3),
#     # MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
#     Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
#     # GlobalAveragePooling2D(),
#     Flatten(),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.3),
#     Dense(4, activation='softmax')
#   ])
#     return model

def build_model(base_model):
    """
    Create a model using a pretrained base model (DenseNet) and add a custom classifier.
    """
    model = Sequential([
    Input(shape=(224,224,3)),
    base_model,
    # Dropout(0.3),
    # MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.0)),
    Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.0)),
    GlobalAveragePooling2D(),
    Dropout(0.3),
    # Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0)),
    
    Dense(4, activation='softmax')
  ])
    return model



@register_keras_serializable()
def binary_crossentropy(y_true, y_pred, pos_weight=15):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = - (pos_weight * y_true * tf.math.log(y_pred) +
              (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

@register_keras_serializable()
def iou_metric(y_true, y_pred, pos_weight=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection / (union + K.epsilon())) * pos_weight


@register_keras_serializable()
def dice_coefficient(y_true, y_pred, pos_weight=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2 * intersection + K.epsilon()) / \
           (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + K.epsilon())
    return dice * pos_weight

# Helper function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def classify_and_segment(file,
                        #  segmented_model = segmented_model,
                        # classificaion_model = classificaion_model,
                        class_names =  ['No Tumor','Glioma', 'Meningioma', 'Pituitary']
                         ):
    global classificaion_model, segmented_model
    # Read the file into memory
    in_memory_file = file.read()
    # Convert the byte data to a numpy array
    nparr = np.frombuffer(in_memory_file, np.uint8)
    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_size = img.shape
    img = enhance_image(img)

    pred = classificaion_model.predict(np.expand_dims(img, axis=0))
    max_id = np.argmax(pred,axis=-1)[0]
    tumor_type = class_names[max_id]
    confidence = round(pred[0,max_id] * 100,2)

    semgnted_img = segmented_model.predict(np.expand_dims(cv2.resize(img, (256, 256)), axis=0))
    print(pred, semgnted_img.shape)
    img = (cv2.resize(semgnted_img[0], (img_size[1], img_size[0])) * 255).astype(np.uint8)
    os.makedirs('temp',exist_ok=True)
    cv2.imwrite('temp/segmented_img.png', img) 
    return tumor_type, float(confidence), 'http://localhost:5000/temp/segmented_img.png'


@app.route('/temp/<filename>')
def get_temp_file(filename):
    return send_from_directory(TEMP_DIR, filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

# Route for image analysis
@app.route('/analyze', methods=['POST'])
def analyze_image():

    if 'originalImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['originalImage']

    tumor_type, confidence, segmented_image = classify_and_segment(file)

    app.config['SEGMENTED_IMAGE'] = segmented_image
    # Return results
    return jsonify({
        'tumorType': tumor_type,
        'tumorConf': confidence,
        # 'segmentedImage': url_for('get_segmented_image', _external=True),  # URL to access the segmented image
        'segmentedImage': segmented_image
    })

@app.route('/get-segmented-image', methods=['GET'])
def get_segmented_image():
    segmented_image = app.config.get('SEGMENTED_IMAGE')
    return send_file(segmented_image, mimetype='image/png', 
                     as_attachment=True, download_name='segmented_image.png')

def enhance_image(image):

    """
    CLAHE and Histogram Equalization: Crucial for medical images with subtle variations in intensity,
    helping make those differences more visible for the model. This helps highlight small details,
    such as edges of tumors, which are often crucial in medical images like MRI scans.

    Gaussian Blurring: Helps smooth the image, reducing distractions like noise while retaining important features.
    Helps in removing unnecessary noise while maintaining the overall structure, making the tumor boundaries more visible.

    Normalization: Ensures that input values are consistent, which helps neural networks converge faster during training.
    """

     # Convert image to uint8 if it is not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE - Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    img_clahe = clahe.apply(img_gray)

    # Gaussian Blurring
    img_blurred = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Convert back to BGR
    img_bgr = cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR)

    # Resize the image to a fixed size (224x224)
    img_resized = cv2.resize(img_bgr, (224, 224))

    # Normalize the image (convert values to the range [0, 1])
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized



# Load existing JSON data
def load_json():
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []  # If file corrupted, start fresh
    return []

# Save updated JSON data
def save_json(data):
    with open(JSON_PATH, 'w') as file:
        json.dump(data, file, indent=4)

@app.route('/save-results', methods=['POST'])
def save_results():
    try:
        # Extract form data
        patient_name = request.form['patientName']
        tumor_type = request.form['tumorType']
        confidence = request.form['confidence']
        other_details = request.form['otherDetails']

        # Extract images
        original_image = request.files['originalImage']
        segmented_image = request.files['segmentedImage']

        # Create unique filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{patient_name}_{timestamp_str}.png'

        original_path = os.path.join(ORIGINAL_IMAGES_DIR, file_name)
        segmented_path = os.path.join(SEGMENTED_IMAGES_DIR, file_name)

        original_image.save(original_path)
        segmented_image.save(segmented_path)

        # Load previous results
        results = load_json()

        # Create new record
        new_record = {
            "Patient Name": patient_name,
            "Tumor Type": tumor_type,
            "Confidence": confidence,
            "Other Details": other_details,
            "Original Image": original_path,
            "Segmented Image": segmented_path,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S")
        }

        # Append and save JSON
        results.append(new_record)
        save_json(results)

        return jsonify({"message": "Results saved successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Create models using DenseNet as the base
classificaion_model = DenseNet121(include_top=False,
                                  classes = 4,
                                  weights=None, #pretrained weights
                                  input_shape=(224, 224, 3)) # input shape of the model

classificaion_model = build_model(classificaion_model)
classificaion_model.load_weights('models/classification.weights.h5')
# compile_model(classificaion_model)
# DenseNet_pretrained.summary()

segmented_model = load_model(
    'models/segmentation.keras',
    custom_objects={
        "binary_crossentropy": binary_crossentropy,
        "iou_metric": iou_metric,
        "dice_coefficient": dice_coefficient
    }
)

@app.route('/get-patient-data')
def get_patient_data():
    records = []
    for filename in os.listdir('data'):
        if filename.endswith(".json"):
            with open(os.path.join("data", filename), 'r') as f:
                records.append(json.load(f))

    return jsonify({"records": records}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

