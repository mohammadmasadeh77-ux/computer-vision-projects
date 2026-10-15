"""Virtual painter: draw in the air with your finger.

- Index + middle fingers up -> selection mode (move up to the header to pick a color)
- Index finger only         -> drawing mode
- Black color               -> eraser
Press 'q' to quit.

Put four header images (brush colors + eraser) in the `header_images/` folder,
named so they sort in order. Each one is resized to 1280x125.

Usage:  python virtual_painter.py [header_folder]
"""

import os
import sys

import cv2
import numpy as np

from hand_tracking_module import HandTrackingModule

BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50
FRAME_W, FRAME_H = 1280, 720
HEADER_H = 125
DEFAULT_FOLDER = "header_images"

# Color per header image, in file-name order: (B, G, R). Black erases.
COLORS = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]


def load_headers(folder):
    headers = []
    for name in sorted(os.listdir(folder)):
        image = cv2.imread(os.path.join(folder, name))
        if image is not None:
            headers.append(cv2.resize(image, (FRAME_W, HEADER_H)))
    return headers


def pick_color_index(x):
    """Return which header button the x position is over, or None."""
    if 250 < x < 450:
        return 0
    if 550 < x < 750:
        return 1
    if 800 < x < 950:
        return 2
    if 1050 < x < 1200:
        return 3
    return None


def main(folder):
    headers = load_headers(folder)
    if len(headers) < len(COLORS):
        raise SystemExit(f"Need {len(COLORS)} header images in '{folder}'.")

    header = headers[0]
    draw_color = COLORS[0]
    xp, yp = 0, 0
    canvas = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_W)
    cap.set(4, FRAME_H)
    detector = HandTrackingModule(min_detection_confidence=0.85)

    while True:
        success, img = cap.read()
        if not success:
            print("Camera not found")
            break

        img = cv2.flip(img, 1)
        # Make sure the frame matches the canvas even if the camera ignores the size request.
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        img = detector.find_hands(img)
        lm_list = detector.findpoints(img, draw=False)

        if lm_list:
            x1, y1 = lm_list[8][1:]    # index fingertip
            x2, y2 = lm_list[12][1:]   # middle fingertip
            fingers = detector.fingersUp()

            # Selection mode: two fingers up.
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < HEADER_H:
                    index = pick_color_index(x1)
                    if index is not None:
                        header = headers[index]
                        draw_color = COLORS[index]
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

            # Drawing mode: only the index finger is up.
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
                cv2.line(img, (xp, yp), (x1, y1), draw_color, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1

        # Merge the canvas into the camera frame.
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inverse = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inverse = cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inverse)
        img = cv2.bitwise_or(img, canvas)

        img[0:HEADER_H, 0:FRAME_W] = header

        cv2.imshow("Virtual Painter", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER)
