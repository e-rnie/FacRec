#!/usr/bin/env python3
# LBPH face‑recognition demo – 100 % OpenCV, no ffmpeg, no libcamera‑vid
import cv2, os, numpy as np

# ---------- training -------------------------------------------------
faces_dir = "faces"          # one sub‑folder or multiple JPGs per person
labels, images = [], []
for label, fname in enumerate(os.listdir(faces_dir)):
    img = cv2.imread(os.path.join(faces_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        images.append(img)
        labels.append(label)
names = [os.path.splitext(f)[0] for f in os.listdir(faces_dir)]

model = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16)
model.train(images, np.array(labels))

# ---------- live camera ----------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # works with libcamera stack
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, conf = model.predict(cv2.resize(roi, (200,200)))
        name = names[label] if conf < 60 else "Unknown"
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.0f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("LBPH preview – ESC to quit", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
