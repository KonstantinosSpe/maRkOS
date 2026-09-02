"""Opening the webcam the way every tool in this project wants it."""

import cv2


def open_camera():
    """The default camera as 640x480 MJPEG (the resolution the calibration and the stars' templates were made at)."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap
