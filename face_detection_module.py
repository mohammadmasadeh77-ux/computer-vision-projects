"""Face detection helper built on MediaPipe Face Detection.

Run this file directly to try it on your webcam.
"""

import time

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.7, model_selection=1):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )
        self.results = None

    def find_faces(self, img, draw=True):
        """Return ([id, (x, y, w, h), score], img) for every face found."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxes = []

        if self.results.detections:
            ih, iw, _ = img.shape
            for face_id, detection in enumerate(self.results.detections):
                rel = detection.location_data.relative_bounding_box
                box = (
                    int(rel.xmin * iw),
                    int(rel.ymin * ih),
                    int(rel.width * iw),
                    int(rel.height * ih),
                )
                bboxes.append([face_id, box, detection.score])

                if draw:
                    img = self.fancy_draw(img, box)
                    score = int(detection.score[0] * 100)
                    cv2.putText(
                        img, f"{score}%", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2,
                    )
        return bboxes, img

    def fancy_draw(self, img, bbox, length=30, thickness=5, rect_thickness=1):
        """Draw a thin box with thick corner marks."""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        color = (255, 0, 255)

        cv2.rectangle(img, (x, y), (x1, y1), color, rect_thickness)

        # Top left, top right, bottom left, bottom right corners.
        for cx, cy, dx, dy in (
            (x, y, 1, 1),
            (x1, y, -1, 1),
            (x, y1, 1, -1),
            (x1, y1, -1, -1),
        ):
            cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
            cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)
        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        _, img = detector.find_faces(img)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
