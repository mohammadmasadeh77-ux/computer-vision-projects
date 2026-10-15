"""Hand tracking helper built on MediaPipe Hands.

Used by the virtual mouse, volume control, finger counter and virtual painter.
"""

import math
import time

import cv2
import mediapipe as mp


class HandTrackingModule:
    # Landmark ids of the five fingertips: thumb, index, middle, ring, pinky.
    TIP_IDS = (4, 8, 12, 16, 20)

    def __init__(
        self,
        mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.lmlist = []

    def find_hands(self, img, draw=True):
        """Detect hands in a BGR frame and optionally draw the skeleton."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def findpoints(self, img, handNo=0, draw=True):
        """Return a list of [id, x, y] pixel landmarks for one hand."""
        self.lmlist = []
        if (
            self.results
            and self.results.multi_hand_landmarks
            and len(self.results.multi_hand_landmarks) > handNo
        ):
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for lm_id, lm in enumerate(hand.landmark):
                xc, yc = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([lm_id, xc, yc])
                if draw:
                    cv2.circle(img, (xc, yc), 5, (255, 0, 250), cv2.FILLED)
        return self.lmlist

    def fingersUp(self):
        """Return [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down).

        The thumb check compares x positions, so it assumes the frame was
        mirrored (cv2.flip) and a right hand is shown.
        """
        if not self.lmlist:
            return []

        fingers = []
        thumb_tip = self.TIP_IDS[0]
        fingers.append(1 if self.lmlist[thumb_tip][1] > self.lmlist[thumb_tip - 1][1] else 0)

        for tip in self.TIP_IDS[1:]:
            fingers.append(1 if self.lmlist[tip][2] < self.lmlist[tip - 2][2] else 0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        """Distance in pixels between two landmarks, plus their coordinates."""
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        lmlist = detector.findpoints(img)

        if lmlist:
            fingers = detector.fingersUp()
            if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                length, img, _ = detector.findDistance(8, 12, img)
                print(length)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
