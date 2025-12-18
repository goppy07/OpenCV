import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
w, h = 770, 480

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

last_update = time.time()
update_delay = 0.5
smoothed_prob = 0.0

def findFace(img):
    global last_update, smoothed_prob

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces, rejectLevels, levelWeights = faceCascade.detectMultiScale3(
        imgGray, 1.2, 5, outputRejectLevels=True
    )

    if len(faces) > 0:

        areas = [w*h for (x,y,w,h) in faces]
        i = areas.index(max(areas))

        x, y, wF, hF = faces[i]
        cx = x + wF // 2
        cy = y + hF // 2

        prob = levelWeights[i] if np.isscalar(levelWeights[i]) else levelWeights[i][0]

        prob_norm = min(prob / 10.0, 1.0)

 
        if time.time() - last_update > 0.5:
            smoothed_prob = prob_norm
            last_update = time.time()

        cv2.rectangle(img, (x, y), (x + wF, y + hF), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        cv2.putText(img, f"P: {smoothed_prob*100:.1f}%", (cx-50, cy-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (w, h))
    frame = findFace(frame)

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
