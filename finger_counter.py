"""Finger counter.

Counts the raised fingers and shows a matching overlay image in the top-left corner.
Put your images in the `finger_images/` folder, named so they sort in order
(for example 1.png ... 5.png, plus one for zero fingers).
Press 'q' to quit.

Usage:  python finger_counter.py [images_folder]
"""

import os
import sys
import time

import cv2

from hand_tracking_module import HandTrackingModule

DEFAULT_FOLDER = "finger_images"


def load_overlays(folder):
    """Load every image in the folder, sorted by file name."""
    overlays = []
    for name in sorted(os.listdir(folder)):
        image = cv2.imread(os.path.join(folder, name))
        if image is not None:
            overlays.append(image)
    return overlays


def main(folder):
    overlays = load_overlays(folder)
    if not overlays:
        raise SystemExit(f"No images found in '{folder}'.")

    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule(min_detection_confidence=0.7)
    p_time = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        lm_list = detector.findpoints(img)

        if lm_list:
            fingers = detector.fingersUp()
            total = fingers.count(1)

            # Index -1 (zero fingers) picks the last image, as in the original design.
            overlay = overlays[(total - 1) % len(overlays)]
            h, w, _ = overlay.shape
            img[0:h, 0:w] = overlay

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (120, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Finger Counter", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER)
