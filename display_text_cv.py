import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
t = 0
start = time.time()
while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    font = cv2.FONT_HERSHEY_SIMPLEX

    if t<10:
        cv2.putText(frame, 'Detecting mood:', (0, height // 2), font, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f'{int(10-t)+1}', (width - 50, height - height // 2), font, 1, (0, 0, 0), 5, cv2.LINE_AA)
    elif t<20:
        cv2.putText(frame, f'QUOTE', (width//2, height // 2), font, 1, (0, 0, 0), 5, cv2.LINE_AA)
    else:
        start = time.time()

    cv2.imshow('frame', frame)




    end = time.time()
    t = end - start
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()