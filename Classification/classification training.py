import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import cv2
import numpy as np

IMG_SIZE = (224, 224)
BATCH = 16
NUM_CLASSES = 4

train_dir = "/content/train"
val_dir = "/content/val"

# ---------------- Preprocessing Function ---------------- #
def custom_preprocess(img):
    # Convert to grayscale (image comes as uint8 [0,255])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # CLAHE for enhancing contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gaussian Blurring for noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Expand channel dimension (224,224,1)
    gray = gray[..., np.newaxis]

    # Normalize → [0,1]
    gray = gray / 255.0

    return gray

# ---------------- Data Augmentation ---------------- #
train_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=custom_preprocess
)

val_gen = ImageDataGenerator(
    preprocessing_function=custom_preprocess
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',  # Converted manually inside preprocessing
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH,
    class_mode='categorical'
)

# ---------------- Model Builder ---------------- #
def build_model(pretrained=True):
    input_tensor = Input(shape=(224, 224, 1))

    # Convert grayscale -> pseudo RGB for DenseNet
    x = tf.keras.layers.Concatenate()([input_tensor, input_tensor, input_tensor])

    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=x
    )

    base.trainable = True  # Fine-tuning allowed

    x = Conv2D(128, (3, 3), activation='relu')(base.output)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=input_tensor, outputs=outputs)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------- Callbacks ---------------- #
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ---------------- Train Function ---------------- #
def train(model, name):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=callbacks,   # <--- Added here
        verbose=1
    )
    
    model.save(f"{name}.h5")
    print(f"Saved {name}.h5")
    return history


# ------- Train #1: Pretrained DenseNet121 ------- #
model_pretrained = build_model(pretrained=True)
history_pretrained = train(model_pretrained, "DenseNet_Clf_pretrained")

# ------- Train #2: Non-Pretrained DenseNet121 ------- #
model_scratch = build_model(pretrained=False)
history_scratch = train(model_scratch, "DenseNet_Clf_scratch")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# ================= PLOT TRAINING HISTORY ================= #
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14,5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot both models
plot_history(history_pretrained, "DenseNet Pretrained")
plot_history(history_scratch, "DenseNet Scratch")

# ================= CLASSIFICATION REPORT ================= #
def evaluate_model(model, data, name):
    y_true = data.classes
    class_labels = list(data.class_indices.keys())

    # Predictions
    pred = model.predict(data)
    y_pred = np.argmax(pred, axis=1)

    print(f"\n========= {name} Classification Report =========")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate both models
evaluate_model(model_pretrained, val_data, "DenseNet Pretrained")
evaluate_model(model_scratch, val_data, "DenseNet Scratch")
