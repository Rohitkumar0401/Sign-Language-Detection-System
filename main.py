import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
INPUT_SIZE = 63

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.keras")
LABEL_PATH = os.path.join(BASE_DIR, "labels.pkl")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    features = []

    for lm in hand_landmarks.landmark:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    return np.array(features, dtype=np.float32)


def load_external_data(dataset_path):
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(LABELS)}

    hands = mp_hands.Hands(static_image_mode=True)

    for label in LABELS:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    features = extract_landmarks(hand_lm)
                    X.append(features)
                    y.append(label_map[label])

    hands.close()
    return np.array(X), np.array(y)


def build_model():
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    print("Loading dataset...")
    X, y = load_external_data("dataset")

    if len(X) == 0:
        print("No data found!")
        return

    with open(LABEL_PATH, "wb") as f:
        pickle.dump(LABELS, f)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        batch_size=32
    )

    model.save(MODEL_PATH)
    print("Model trained and saved!")


def detect():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        print("Train model first!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_PATH, "rb") as f:
        labels = pickle.load(f)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    smooth_label = ""
    confidence_threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        display_text = "No Hand"

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                features = extract_landmarks(hand_lm).reshape(1, -1)
                pred = model.predict(features, verbose=0)

                idx = np.argmax(pred)
                confidence = np.max(pred)

                if confidence > confidence_threshold:
                    smooth_label = labels[idx]
                    display_text = f"{smooth_label} ({confidence*100:.1f}%)"
                else:
                    display_text = "Low Confidence"

        cv2.putText(
            frame,
            display_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow("Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "detect"], default="detect")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    else:
        detect()