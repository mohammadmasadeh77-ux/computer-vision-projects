"""Gesture-controlled virtual mouse.

- Index finger up only        -> move the cursor
- Index + middle fingers up   -> bring them together (pinch) to click
Press 'q' to quit.
"""

import time

import cv2
import numpy as np
import pyautogui

from hand_tracking_module import HandTrackingModule

BOX_W, BOX_H = 400, 225   # active area inside the camera frame
SMOOTHENING = 5           # higher = smoother but slower cursor
CLICK_DISTANCE = 40       # pixels between index and middle fingertips

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def main():
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule(max_num_hands=1)
    screen_w, screen_h = pyautogui.size()

    p_time = 0
    ploc_x, ploc_y = 0, 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Centered rectangle that maps to the whole screen.
        x_min = (w - BOX_W) // 2
        x_max = x_min + BOX_W
        y_min = (h - BOX_H) // 2
        y_max = y_min + BOX_H
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        img = detector.find_hands(img)
        lm_list = detector.findpoints(img)

        if lm_list:
            x1, y1 = lm_list[8][1:]   # index fingertip
            fingers = detector.fingersUp()

            if len(fingers) >= 3:
                # Moving mode: only the index finger is up.
                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (x_min, x_max), (0, screen_w))
                    y3 = np.interp(y1, (y_min, y_max), (0, screen_h))

                    cloc_x = ploc_x + (x3 - ploc_x) / SMOOTHENING
                    cloc_y = ploc_y + (y3 - ploc_y) / SMOOTHENING

                    pyautogui.moveTo(cloc_x, cloc_y)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    ploc_x, ploc_y = cloc_x, cloc_y

                # Clicking mode: index and middle fingers are up and close together.
                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, _ = detector.findDistance(8, 12, img)
                    if length < CLICK_DISTANCE:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
