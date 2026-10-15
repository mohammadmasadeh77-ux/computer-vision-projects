"""AI gym repetition counter.

Tracks the right arm (shoulder 12, elbow 14, wrist 16), converts the elbow angle
into a 0-100 % progress value, and counts one repetition for every full
up-and-down movement. Press 'q' to quit.

Usage:  python gym_rep_counter.py path/to/video.mp4
        python gym_rep_counter.py            (uses the webcam)
"""

import sys
import time

import cv2
import numpy as np

from pose_module import PoseDetector

ANGLE_RANGE = (45, 145)   # elbow angle (degrees) that maps to 100 % ... 0 %
SHOULDER, ELBOW, WRIST = 12, 14, 16


def main(source):
    cap = cv2.VideoCapture(source)
    detector = PoseDetector()

    p_time = 0
    direction = 0   # 0 = going up, 1 = going down
    count = 0.0     # each half movement adds 0.5

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1280, 720))
        img = detector.find_pose(img, draw=False)
        lm_list = detector.get_position(img, draw=False)

        if lm_list:
            angle = detector.find_angle(img, SHOULDER, ELBOW, WRIST)
            per = float(np.clip(np.interp(angle, ANGLE_RANGE, (100, 0)), 0, 100))

            if per == 100 and direction == 0:
                count += 0.5
                direction = 1
            if per == 0 and direction == 1:
                count += 0.5
                direction = 0

            print(f"Angle: {int(angle)} | Progress: {int(per)}% | Count: {int(count)}")

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, f"Count: {int(count)}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)

        cv2.imshow("Gym Rep Counter", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 0)
