# Computer Vision Projects

Real-time computer vision projects built with Python, OpenCV, and MediaPipe.
Everything runs from your webcam (the gym counter can also use a video file), and you press `q` to quit.

| Project | File | What it does |
|---|---|---|
| Virtual mouse | `virtual_mouse.py` | Move the cursor with your index finger and click with a pinch gesture. Motion smoothing reduces jitter. |
| Volume control | `volume_control.py` | Set the Windows system volume with the distance between thumb and index finger. Windows only. |
| Finger counter | `finger_counter.py` | Counts raised fingers and shows a matching overlay image. |
| Virtual painter | `virtual_painter.py` | Draw in the air, pick colors from an on-screen header, and erase, all with finger gestures. |
| Gym rep counter | `gym_rep_counter.py` | Computes the elbow angle from pose landmarks and counts repetitions automatically. |

## Reusable modules

| Module | Purpose |
|---|---|
| `hand_tracking_module.py` | Hand landmarks, fingers-up detection, and distance between landmarks. |
| `pose_module.py` | Body pose landmarks and joint-angle calculation. |
| `face_detection_module.py` | Face detection with custom bounding boxes and confidence scores. |

Each module can be run on its own as a quick demo, for example `python hand_tracking_module.py`.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

> MediaPipe 0.10.30 and newer removed the `mp.solutions` API used here, so `requirements.txt` pins an older release. That release supports Python 3.9 to 3.12.

## Run

```bash
python virtual_mouse.py
python volume_control.py          # Windows only
python finger_counter.py          # needs images in finger_images/
python virtual_painter.py         # needs 4 header images in header_images/
python gym_rep_counter.py my_video.mp4
```

### Images and videos

The finger counter and the painter need their own images, which are not included in this repository:

- `finger_images/`: one image per finger count, named so they sort in order (for example `1.png` to `5.png`, plus one for zero fingers).
- `header_images/`: four header images (three brush colors and an eraser), named so they sort in order. They are resized to 1280x125.

The gym counter takes the path of your own video as an argument.

## Notes

- The thumb check in `fingersUp()` compares x positions, so it assumes a mirrored frame and a right hand.
- The gym counter tracks the right arm (landmarks 12, 14, 16). Change the constants at the top of the file to track another joint.

## Tech

Python, OpenCV, MediaPipe, NumPy, PyAutoGUI, pycaw
