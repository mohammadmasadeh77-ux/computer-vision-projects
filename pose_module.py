"""Pose estimation helper built on MediaPipe Pose.

Used by the AI gym repetition counter.
Run this file directly to try it on a video:  python pose_module.py path/to/video.mp4
"""

import math
import sys
import time

import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(
        self,
        mode=False,
        complexity=1,
        smooth=True,
        detection_con=0.5,
        track_con=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            smooth_landmarks=smooth,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.lm_list = []

    def find_pose(self, img, draw=True):
        """Detect the body pose in a BGR frame and optionally draw it."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def get_position(self, img, draw=True):
        """Return a list of [id, x, y] pixel landmarks (empty if no body found)."""
        self.lm_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """Angle in degrees (0-180) at landmark p2, between p1 and p3."""
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            for x, y in ((x1, y1), (x2, y2), (x3, y3)):
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            cv2.putText(
                img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2,
            )
        return angle


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        lm_list = detector.get_position(img, draw=False)
        if lm_list:
            # Highlight the right elbow (landmark 14).
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Pose Estimation", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Pass a video path, or use the webcam when no argument is given.
    main(sys.argv[1] if len(sys.argv) > 1 else 0)
