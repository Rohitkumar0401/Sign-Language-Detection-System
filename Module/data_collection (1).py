"""
================================================
  SIGN LANGUAGE DETECTION SYSTEM
  Phase 1: Data Collection
  Author: Custom ASL Dataset Trainer
================================================

HOW TO USE:
- Press a LETTER KEY (A-Z) to start capturing images for that class
- Press 'S' to STOP capturing
- Press 'Q' to QUIT
- Images are saved to: dataset/<LETTER>/

NOTE: If you already have asl_alphabet_train dataset,
      run train_model.py directly. This script is for
      collecting your own custom hand sign images.
"""

import cv2
import mediapipe as mp
import os
import time

# ── Settings ──────────────────────────────────────────────
DATASET_DIR = "dataset"
IMG_SIZE = 128          # pixels (width & height)
IMAGES_PER_CLASS = 300  # target images per letter
CAPTURE_DELAY = 0.05    # seconds between captures (~20 fps)
PADDING = 30            # pixels padding around hand crop
# ──────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.7)

os.makedirs(DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam. Check camera connection.")

current_label   = None
capturing       = False
last_capture_t  = 0
count           = 0

print("\n🟢 Data Collection Started")
print("   Press a LETTER KEY to begin capturing for that class.")
print("   Press 'S' to stop.  Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_crop = None

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Compute bounding box from landmarks
            xs = [lm.x * w for lm in hand_lm.landmark]
            ys = [lm.y * h for lm in hand_lm.landmark]
            x1 = max(0, int(min(xs)) - PADDING)
            y1 = max(0, int(min(ys)) - PADDING)
            x2 = min(w, int(max(xs)) + PADDING)
            y2 = min(h, int(max(ys)) + PADDING)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hand_crop = frame[y1:y2, x1:x2]

    # ── Auto-capture ───────────────────────────────────────
    if capturing and current_label and hand_crop is not None:
        now = time.time()
        if now - last_capture_t >= CAPTURE_DELAY:
            label_dir = os.path.join(DATASET_DIR, current_label)
            os.makedirs(label_dir, exist_ok=True)
            existing = len(os.listdir(label_dir))

            if existing < IMAGES_PER_CLASS:
                img_path = os.path.join(label_dir, f"{existing:04d}.jpg")
                resized  = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(img_path, resized)
                count          = existing + 1
                last_capture_t = now
            else:
                capturing = False
                print(f"✅ Done capturing '{current_label}' — {IMAGES_PER_CLASS} images saved.")

    # ── HUD overlay ───────────────────────────────────────
    status_color = (0, 200, 0) if capturing else (0, 100, 255)
    status_text  = f"CAPTURING: {current_label}  [{count}/{IMAGES_PER_CLASS}]" if capturing else "IDLE — Press a letter key"
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    cv2.imshow("Data Collection — Sign Language", frame)

    # ── Key handling ──────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Exiting data collection.")
        break
    elif key == ord('s') or key == ord('S'):
        capturing = False
        print(f"⏹  Stopped capturing. '{current_label}' has {count} images.")
    elif 65 <= key <= 90 or 97 <= key <= 122:   # A-Z or a-z
        letter = chr(key).upper()
        label_dir = os.path.join(DATASET_DIR, letter)
        os.makedirs(label_dir, exist_ok=True)
        count         = len(os.listdir(label_dir))
        current_label = letter
        capturing     = True
        print(f"📸 Capturing for class '{letter}' (already have {count} images) …")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Data collection complete. Run train_model.py next.")
