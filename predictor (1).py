"""
predictor.py
============
Real-time ASL hand-sign predictor.

• MediaPipe extracts hand landmarks and crops the hand ROI.
• The saved Keras model classifies the crop.
• A rolling-window majority vote smooths jittery predictions.
• pyttsx3 speaks each newly confirmed letter (non-blocking thread).
"""

import os
import threading
import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np

# ── Optional TF import (needed only when running as a standalone module) ──────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

IMG_SIZE = 96           # must match training size (train_model.py se same)
SMOOTH_WINDOW  = 15     # frames for majority-vote smoothing
CONFIDENCE_THR = 0.65   # minimum confidence to accept a prediction
ROI_PADDING    = 30     # extra pixels around hand bounding box


class TTSSpeaker:
    """Thread-safe, non-blocking text-to-speech wrapper."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_spoken: str = ""
        self._busy = False
        if _TTS_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
                self._engine.setProperty("volume", 0.9)
            except Exception:
                self._engine = None
        else:
            self._engine = None

    def speak(self, text: str):
        """Speak *text* only if it differs from the last spoken word."""
        if text == self._last_spoken or self._busy:
            return
        self._last_spoken = text
        if self._engine is None:
            return
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()

    def _run(self, text: str):
        with self._lock:
            self._busy = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass
            finally:
                self._busy = False


class ASLPredictor:
    """
    Wraps model loading, MediaPipe hand detection, ROI cropping,
    prediction smoothing, and TTS into a single reusable object.

    Usage
    -----
    predictor = ASLPredictor(model_path, class_indices_path)
    frame_bgr, letter, confidence = predictor.process(frame_bgr)
    """

    # ── Colours & fonts ───────────────────────────────────────────────────────
    _LABEL_BG   = (0, 200, 80)
    _LABEL_FG   = (255, 255, 255)
    _BOX_COLOR  = (0, 230, 120)
    _FONT       = cv2.FONT_HERSHEY_DUPLEX
    _FONT_SCALE = 1.8
    _THICKNESS  = 3

    def __init__(
        self,
        model_path: str = "models/asl_mobilenetv2.h5",
        class_idx_path: str = "models/class_indices.npy",
    ):
        # ── Load model ────────────────────────────────────────────────────────
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run train_model.py first."
            )
        self.model = tf.keras.models.load_model(model_path)

        # ── Load class index mapping ──────────────────────────────────────────
        raw = np.load(class_idx_path, allow_pickle=True).item()
        # Invert: index → label
        self.idx_to_label: dict[int, str] = {v: k for k, v in raw.items()}

        # ── MediaPipe Hands ───────────────────────────────────────────────────
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )

        # ── Smoothing ─────────────────────────────────────────────────────────
        self._vote_window: deque[str] = deque(maxlen=SMOOTH_WINDOW)
        self._stable_letter: str = ""
        self._stable_conf: float = 0.0

        # ── TTS ───────────────────────────────────────────────────────────────
        self.speaker = TTSSpeaker()

        print("✅  ASLPredictor ready.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise an ROI for model inference."""
        roi = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)

    def _get_hand_roi(self, frame: np.ndarray, landmarks) -> np.ndarray | None:
        """Crop the hand region with padding; returns None if out of bounds."""
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        x1 = max(0, int(min(xs)) - ROI_PADDING)
        y1 = max(0, int(min(ys)) - ROI_PADDING)
        x2 = min(w, int(max(xs)) + ROI_PADDING)
        y2 = min(h, int(max(ys)) + ROI_PADDING)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _smooth(self, letter: str) -> str:
        """Add *letter* to vote window; return majority vote."""
        self._vote_window.append(letter)
        majority, _ = Counter(self._vote_window).most_common(1)[0]
        return majority

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, str, float]:
        """
        Detect hand, run model, overlay result.

        Returns
        -------
        annotated_frame : np.ndarray   BGR frame with overlaid text & box
        letter          : str          Predicted letter ('' if no hand)
        confidence      : float        Confidence in [0, 1]
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        letter, confidence = "", 0.0

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                # Draw skeleton
                self._mp_draw.draw_landmarks(
                    frame_bgr, hand_lm, self._mp_hands.HAND_CONNECTIONS
                )

                # Crop ROI
                roi_result = self._get_hand_roi(frame_bgr, hand_lm)
                if roi_result is None:
                    continue
                roi, (x1, y1, x2, y2) = roi_result

                # Predict
                inp  = self._preprocess(roi)
                pred = self.model.predict(inp, verbose=0)[0]
                idx  = int(np.argmax(pred))
                raw_conf = float(pred[idx])
                raw_letter = self.idx_to_label.get(idx, "?")

                # Only accept confident predictions
                if raw_conf >= CONFIDENCE_THR:
                    smoothed = self._smooth(raw_letter)
                else:
                    smoothed = self._smooth("")   # push empty to window
                    raw_conf = 0.0

                confidence = raw_conf
                letter     = smoothed

                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), self._BOX_COLOR, 2)

                # Draw label above box
                if letter:
                    label = f"{letter}  {int(confidence * 100)}%"
                    (tw, th), _ = cv2.getTextSize(
                        label, self._FONT, self._FONT_SCALE, self._THICKNESS
                    )
                    lx, ly = x1, max(y1 - 10, th + 10)
                    cv2.rectangle(
                        frame_bgr,
                        (lx, ly - th - 10),
                        (lx + tw + 10, ly + 4),
                        self._LABEL_BG, -1,
                    )
                    cv2.putText(
                        frame_bgr, label,
                        (lx + 4, ly),
                        self._FONT, self._FONT_SCALE,
                        self._LABEL_FG, self._THICKNESS, cv2.LINE_AA,
                    )

                    # Speak only newly stable letters
                    if letter != self._stable_letter:
                        self._stable_letter = letter
                        self._stable_conf   = confidence
                        self.speaker.speak(letter)

        else:
            # No hand detected — clear state
            self._vote_window.clear()
            self._stable_letter = ""

        return frame_bgr, letter, confidence

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


# ── Standalone webcam demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = ASLPredictor()
    cap = cv2.VideoCapture(0)

    print("Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)                    # mirror
        annotated, letter, conf = predictor.process(frame)
        cv2.imshow("ASL Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    predictor.release()
