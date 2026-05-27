"""
train_model.py
==============
Trains a MobileNetV2-based CNN on the ASL Alphabet dataset.
Optimized for fast training on CPU.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Speed Optimizations ───────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # TF warnings band karo
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE       = 96           # 160 se 96 → bahut fast hoga, accuracy 90-94%
NUM_CLASSES    = 29
MODEL_PATH     = "models/asl_mobilenetv2.h5"
CLASS_IDX_PATH = "models/class_indices.npy"

# ── Default Paths ─────────────────────────────────────────────────────────────
DEFAULT_TRAIN_DIR = r"D:\Sing_language_detection_model\files\Sign_language\detection\asl_alphabet_train"
DEFAULT_TEST_DIR  = r"D:\Sing_language_detection_model\files\Sign_language\detection\asl_alphabet_test"


# ── Data generators ───────────────────────────────────────────────────────────

def build_generators(train_dir: str, test_dir: str, batch_size: int):
    """Return (train_gen, val_gen, test_gen) — light augmentation for speed."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,          # 15 se 10 — kam augmentation = fast
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.10,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.10,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_kw = dict(
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        seed=42,
        interpolation="bilinear",   # fast resize method
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir, subset="training", **common_kw
    )
    val_gen = val_datagen.flow_from_directory(
        train_dir, subset="validation", **common_kw
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, shuffle=False, **common_kw
    )

    return train_gen, val_gen, test_gen


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    """MobileNetV2 + lightweight head — fast inference."""

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.75,             # 0.75 = lighter model, 25% faster than alpha=1.0
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),   # 256 se 128 — chota head
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation="softmax"),
    ], name="ASL_MobileNetV2_Fast")

    return model


def compile_model(model: tf.keras.Model, lr: float = 1e-3):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs("models", exist_ok=True)

    print("⟳  Building data generators …")
    print(f"   Train dir  : {args.train_dir}")
    print(f"   Test dir   : {args.test_dir}")
    print(f"   Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch size : {args.batch_size}")
    print(f"   Epochs     : {args.epochs} + 2 fine-tune")

    train_gen, val_gen, test_gen = build_generators(
        args.train_dir, args.test_dir, args.batch_size
    )

    np.save(CLASS_IDX_PATH, train_gen.class_indices)
    print(f"✔  Classes ({len(train_gen.class_indices)}): {list(train_gen.class_indices.keys())}")

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes)
    compile_model(model, lr=1e-3)
    model.summary()

    cb_list = [
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3,   # 5 se 3 — jaldi stop karo
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1    # 3 se 2
        ),
    ]

    # ── Phase 1: Sirf head train karo ────────────────────────────────────────
    print("\n📚 Phase 1 – training head only …")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cb_list,
        workers=4,                  # parallel data loading
        use_multiprocessing=False,  # Windows pe False rakhna
    )

    # ── Phase 2: Thodi fine-tuning ────────────────────────────────────────────
    print("\n🔧 Phase 2 – fine-tuning top 20 layers …")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-20]:   # 40 se 20 — kam layers
        layer.trainable = False

    compile_model(model, lr=1e-5)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        callbacks=cb_list,
        workers=4,
        use_multiprocessing=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n📊 Evaluating on test set …")
    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy : {acc * 100:.2f}%")
    print(f"Test loss     : {loss:.4f}")
    print(f"\n✅  Model saved → {MODEL_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL alphabet model")
    parser.add_argument("--train_dir",  default=DEFAULT_TRAIN_DIR, help="Path to training dataset")
    parser.add_argument("--test_dir",   default=DEFAULT_TEST_DIR,  help="Path to test dataset")
    parser.add_argument("--epochs",     type=int, default=5,        help="Training epochs (phase 1)")
    parser.add_argument("--batch_size", type=int, default=64,       help="Batch size")
    args = parser.parse_args()
    train(args)
