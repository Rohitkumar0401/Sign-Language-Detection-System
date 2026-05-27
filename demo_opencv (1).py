"""
demo_opencv.py
==============
Lightweight standalone demo — no Streamlit/WebRTC needed.
Runs directly with a webcam using plain OpenCV windows.

Run:
    python demo_opencv.py

Press Q or Esc to quit.
"""

import cv2
import time
from predictor import ASLPredictor

# ── Configuration ─────────────────────────────────────────────────────────────
CAM_INDEX    = 0       # 0 = default webcam
WIN_TITLE    = "ASL Sign Language Detector  |  Q to quit"
TARGET_FPS   = 30

# ── HUD overlay settings ──────────────────────────────────────────────────────
HUD_BG       = (10, 20, 40)
HUD_ACCENT   = (0, 220, 140)
HUD_FONT     = cv2.FONT_HERSHEY_DUPLEX
HUD_FONT_SM  = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(frame, letter: str, confidence: float, history: list[str], fps: float):
    """Overlay a translucent HUD on the bottom of the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent bottom bar
    overlay = frame.copy()
    bar_h = 80
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - bar_h + 22),
                HUD_FONT_SM, 0.55, (100, 160, 100), 1, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "Q / Esc — quit", (10, h - 10),
                HUD_FONT_SM, 0.50, (80, 120, 80), 1, cv2.LINE_AA)

    # History strip (recent letters)
    hist_str = " ".join(history[-12:])
    cv2.putText(frame, f"History: {hist_str}", (w // 2 - 160, h - 10),
                HUD_FONT_SM, 0.55, (150, 200, 255), 1, cv2.LINE_AA)

    # Confidence bar (top-right corner)
    bar_x, bar_y, bar_w, bar_bh = w - 180, 12, 160, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_bh), (30, 50, 80), -1)
    fill_w = int(bar_w * confidence)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_bh), HUD_ACCENT, -1)
    cv2.putText(frame, f"Conf {int(confidence*100)}%", (bar_x, bar_y + bar_bh + 16),
                HUD_FONT_SM, 0.50, HUD_ACCENT, 1, cv2.LINE_AA)

    return frame


def main():
    predictor = ASLPredictor()
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"❌  Could not open camera index {CAM_INDEX}.")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    history: list[str] = []
    prev_letter = ""
    frame_times = []

    print(f"📷  Camera opened.  {WIN_TITLE}")
    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame grab failed; retrying…")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)                          # mirror view

        # ── Run predictor ─────────────────────────────────────────────────────
        annotated, letter, confidence = predictor.process(frame)

        # Update history
        if letter and letter != prev_letter:
            history.append(letter)
            if len(history) > 50:
                history = history[-50:]
        prev_letter = letter

        # ── FPS calculation ───────────────────────────────────────────────────
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        # ── HUD overlay ───────────────────────────────────────────────────────
        draw_hud(annotated, letter, confidence, history, fps)

        cv2.imshow(WIN_TITLE, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    predictor.release()
    print("\nBye! 👋")


if __name__ == "__main__":
    main()
