"""Hand-gesture volume control (Windows only).

The distance between the thumb tip and the index fingertip sets the system volume.
Press 'q' to quit.
"""

import math
import time

import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities

from hand_tracking_module import HandTrackingModule

MIN_LENGTH, MAX_LENGTH = 50, 270   # pixel distance range that maps to 0-100 %


def get_volume_interface():
    """Return the Windows master volume interface (works across pycaw versions)."""
    devices = AudioUtilities.GetSpeakers()
    try:
        return devices.EndpointVolume
    except AttributeError:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume

        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))


def main():
    volume = get_volume_interface()
    min_vol, max_vol = volume.GetVolumeRange()[:2]

    vol_bar, vol_per = 400, 0
    p_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule(min_detection_confidence=0.7)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        lm_list = detector.findpoints(img)

        if lm_list:
            x1, y1 = lm_list[4][1], lm_list[4][2]   # thumb tip
            x2, y2 = lm_list[8][1], lm_list[8][2]   # index fingertip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < MIN_LENGTH:
                cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

            vol = np.interp(length, [MIN_LENGTH, MAX_LENGTH], [min_vol, max_vol])
            vol_bar = np.interp(length, [MIN_LENGTH, MAX_LENGTH], [400, 150])
            vol_per = np.interp(length, [MIN_LENGTH, MAX_LENGTH], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

        # Volume bar.
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{int(vol_per)} %", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Control Volume", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
