

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import threading
import sys
import os

MODEL_PATH        = "model.h5"
LABEL_MAP_PATH    = "label_map.npy"
IMG_SIZE          = 64
CONFIDENCE_THRESH = 0.80
STABLE_FRAMES     = 10
SPEAK_COOLDOWN    = 2.0


ROI_TOP    = 100
ROI_BOTTOM = 400
ROI_LEFT   = 300
ROI_RIGHT  = 600
# ──────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"'{MODEL_PATH}' not found. Run train_model.py first.")
    sys.exit(1)
if not os.path.exists(LABEL_MAP_PATH):
    print(f"'{LABEL_MAP_PATH}' not found. Run train_model.py first.")
    sys.exit(1)

print("🔄 Loading model ...")
model   = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(LABEL_MAP_PATH, allow_pickle=True)
print(f"✅ Model loaded. Classes: {list(classes)}\n")

tts_engine   = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_engine.setProperty('volume', 0.9)
tts_enabled  = True
last_spoken  = ""
last_speak_t = 0.0

def speak_async(text):
    def _run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot open webcam.")
    sys.exit(1)

sentence     = ""
prev_label   = None
stable_count = 0
paused       = False
fps_list     = []
frame_time   = time.time()

C_GREEN = (80,  220, 100)
C_BLUE  = (255, 180, 50)
C_WHITE = (240, 240, 240)
C_GRAY  = (120, 120, 120)
C_RED   = (60,  60,  220)

def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

print(" Real-time detection started.")
print("   Place your hand inside the GREEN BOX.")
print("   Q=Quit  SPACE=Pause  C=Clear  T=TTS\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    now        = time.time()
    fps        = 1.0 / max(now - frame_time, 1e-6)
    frame_time = now
    fps_list.append(fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    predicted_label = None
    confidence      = 0.0

    y1 = max(0, ROI_TOP)
    y2 = min(h, ROI_BOTTOM)
    x1 = max(0, ROI_LEFT)
    x2 = min(w, ROI_RIGHT)

    if not paused:
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size > 0:
            inp        = preprocess(hand_crop)
            pred       = model.predict(inp, verbose=0)[0]
            idx        = np.argmax(pred)
            confidence = float(pred[idx])

            if confidence >= CONFIDENCE_THRESH:
                predicted_label = str(classes[idx])
            else:
                predicted_label = "?"

        if predicted_label and predicted_label != "?":
            if predicted_label == prev_label:
                stable_count += 1
            else:
                stable_count = 0
                prev_label   = predicted_label

            if stable_count == STABLE_FRAMES:
                if predicted_label == "space":
                    sentence += " "
                elif predicted_label == "del":
                    sentence = sentence[:-1]
                else:
                    sentence += predicted_label

                if tts_enabled:
                    t_now = time.time()
                    if predicted_label != last_spoken or t_now - last_speak_t > SPEAK_COOLDOWN:
                        speak_async(predicted_label)
                        last_spoken  = predicted_label
                        last_speak_t = t_now
        else:
            stable_count = 0

    box_color = C_GREEN if (predicted_label and predicted_label != "?") else (0, 150, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_BLUE, 2)
    tts_str = "TTS: ON" if tts_enabled else "TTS: OFF"
    cv2.putText(frame, tts_str, (w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                C_GREEN if tts_enabled else C_GRAY, 2)

    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "PAUSED", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, C_BLUE, 3)

    if predicted_label and not paused:
        conf_pct = int(confidence * 100)
        cv2.rectangle(frame, (0, h - 110), (280, h), (20, 20, 20), -1)
        cv2.putText(frame, "Sign:", (10, h - 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GRAY, 1)
        label_disp = predicted_label if predicted_label != "?" else "Low conf."
        cv2.putText(frame, label_disp, (10, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, C_GREEN, 3)

        bar_x, bar_y, bar_w, bar_bh = 10, h - 15, 200, 10
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_bh), C_GRAY, -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_bh),
                      C_GREEN if confidence >= CONFIDENCE_THRESH else C_RED, -1)
        cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_w + 8, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)

    if not paused:
        ring_frac = stable_count / STABLE_FRAMES
        cv2.circle(frame, (w - 40, h - 40), 22, C_GRAY, 2)
        if ring_frac > 0:
            cv2.ellipse(frame, (w - 40, h - 40), (22, 22),
                        -90, 0, int(360 * ring_frac), C_GREEN, 3)

    panel_y = h - 145
    cv2.rectangle(frame, (0, panel_y), (w, panel_y + 35), (30, 30, 30), -1)
    cv2.putText(frame, "Text: " + sentence[-50:], (10, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2)

    hints = ["Q:Quit", "SPC:Pause", "C:Clear", "T:TTS"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (w - 115, 85 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)

    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == 32:
        paused = not paused
    elif key == ord('c') or key == ord('C'):
        sentence = ""
    elif key == ord('t') or key == ord('T'):
        tts_enabled = not tts_enabled

cap.release()
cv2.destroyAllWindows()
print(f"\n👋 Session ended. Final text: {sentence}")
