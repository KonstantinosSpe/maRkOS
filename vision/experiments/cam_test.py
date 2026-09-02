"""Smoke test: can the default camera be opened and read?"""

import cv2

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if ok:
    print('SUCCESS', frame.shape)
else:
    print('FAILED')
cap.release()
