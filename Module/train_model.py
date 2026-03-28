

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATASET_PATH   = "asl_alphabet_train/asl_alphabet_train"
IMG_SIZE       = 64
BATCH_SIZE     = 32
EPOCHS         = 10
TEST_SPLIT     = 0.2
MODEL_PATH     = "model.h5"
LABEL_MAP_PATH = "label_map.npy"

if not os.path.isdir(DATASET_PATH):
    print(f" Dataset folder not found: '{DATASET_PATH}'")
    print("   Make sure asl_alphabet_train folder is in the same directory as this script.")
    sys.exit(1)

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
print(f" Dataset found: {DATASET_PATH}")
print(f"   Classes ({len(class_names)}): {class_names}\n")

print("Loading images ...")
X, y = [], []

for label in class_names:
    label_dir = os.path.join(DATASET_PATH, label)
    files = [f for f in os.listdir(label_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in files:
        img = cv2.imread(os.path.join(label_dir, fname))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        y.append(label)

    print(f"   {label:>8s}: {len(files)} images")

print(f"\n   Total images loaded: {len(X)}\n")

X = np.array(X, dtype='float32') / 255.0

le = LabelEncoder()
y_enc = le.fit_transform(y)
np.save(LABEL_MAP_PATH, le.classes_)
print(f"💾 Label map saved → {LABEL_MAP_PATH}")
print(f"   Labels: {list(le.classes_)}\n")

num_classes = len(le.classes_)
y_cat = tf.keras.utils.to_categorical(y_enc, num_classes)


X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=TEST_SPLIT,
    random_state=42,
    stratify=y_enc
)
print(f" Train: {len(X_train)}  |  Validation: {len(X_val)}\n")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


cb_list = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                               save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                             restore_best_weights=True, verbose=1),
]

print("\n Training started ...\n")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=cb_list,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n Validation Accuracy : {val_acc*100:.2f}%")
print(f" Validation Loss     : {val_loss:.4f}")
print(f" Model saved        → {MODEL_PATH}\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print(" Training curves saved → training_curves.png")
print("\n Done! Run main.py for real-time detection.")
