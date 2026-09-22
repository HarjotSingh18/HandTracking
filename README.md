# Hand Tracking Project

A real-time hand tracking and gesture recognition project built with Python, OpenCV, and MediaPipe.

## About the Project

This is my first computer vision project. I chose hand tracking as a starting point because MediaPipe makes detecting hand landmarks fast and straightforward, while OpenCV handles the live webcam feed.

Through this project I'm getting hands-on experience with:

- Capturing and processing live video frames in OpenCV
- Extracting 21 hand landmarks per hand in real time
- Converting landmark coordinates into pixel positions
- Building simple gesture recognition from landmark geometry

## Features

- **Real-time hand tracking** for up to two hands, with the landmark skeleton drawn on screen
- **Landmark labels** showing the ID (0–20) of every point on each hand
- **Left/right hand detection**, with each hand labelled on screen
- **Fingertip highlighting** for the pointer finger and thumb, color-coded by hand
- **Gesture recognition:**
  - **Heart:** both pointer fingertips and both thumb tips touching, pointers above thumbs
  - **Thumbs up:** thumb pointing up with the other four fingers curled (works with either hand)
- **FPS counter** in the corner of the window

## Hand Landmark Reference

MediaPipe tracks 21 points on each hand:

| Part    | Landmark IDs | Tip |
|---------|--------------|-----|
| Wrist   | 0            | –   |
| Thumb   | 1–4          | 4   |
| Pointer | 5–8          | 8   |
| Middle  | 9–12         | 12  |
| Ring    | 13–16        | 16  |
| Pinky   | 17–20        | 20  |

## Project Structure

```
HandTracking/
├── handtracking.py         # Basic version: tracks hands and labels landmarks
└── HandTrackingModule.py   # Reusable HandDetector class with gesture detection
```

## Tech Stack

- **Python 3.12**
- **OpenCV**
- **MediaPipe 0.10.21**

## Getting Started

### Prerequisites

This project uses MediaPipe's `solutions` API, which was removed in newer MediaPipe releases. It requires **MediaPipe 0.10.21**, which supports **Python 3.9–3.12** (not 3.13 or newer).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarjotSingh18/HandTracking.git
   cd HandTracking
   ```

2. Create and activate a virtual environment with Python 3.12:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. Install the dependencies:
   ```bash
   pip install mediapipe==0.10.21 opencv-python
   ```

### Running the Project

```bash
python HandTrackingModule.py
```

Press **q** with the video window focused to quit.

### Notes

- **First launch may take a minute.** Matplotlib (used by MediaPipe) builds a font cache the first time it runs. Let it finish; later launches are quick.
- **macOS camera permission.** The first time you run it, macOS will ask to allow camera access for your terminal or code editor.
- **Mirrored view.** The video is flipped horizontally so it behaves like a mirror and MediaPipe's left/right labels match your real hands.
- **Startup log messages** from MediaPipe (`WARNING`, `INFO`, `W0000 ...`) are normal and can be ignored.

## Using the HandDetector Class

`HandTrackingModule.py` can be imported into other projects:

```python
import cv2
from HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    hands = detector.findAllPositions(img)

    for hand in hands:
        if detector.thumbsUp(hand):
            print(f'{hand["type"]} hand: thumbs up!')

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

| Method | Description |
|--------|-------------|
| `findHands(img, draw=True)` | Detects hands in the frame and draws the skeleton. Call this first each frame. |
| `findPosition(img, handNo=0, draw=True)` | Returns `[(id, x, y), ...]` pixel positions for one hand. |
| `findAllPositions(img, draw=True)` | Returns a list of every detected hand as `{"type": "Left"/"Right", "lmList": [...]}`. |
| `heart(right_hand, left_hand)` | Returns `True` if the two hands are making a heart shape. |
| `thumbsUp(hand)` | Returns `True` if the given hand is giving a thumbs up. |

## Future Ideas

- More gestures (peace sign, finger counting, pinch)
- Controlling the mouse or volume with hand movements
- Drawing on screen with the pointer finger
