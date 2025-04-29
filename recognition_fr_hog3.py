#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import numpy as np

# load known face encodings
data = pickle.load(open("encodings.pickle", "rb"))

# open camera at low resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert BGR -> RGB, then detect faces
    rgb = frame[:, :, ::-1]
    # small image = faster, but HOG model works ok at 320×240
    face_locs = face_recognition.face_locations(rgb, model="hog")
    face_encs = face_recognition.face_encodings(
        rgb, known_face_locations=face_locs, num_jitters=1
    )

    # loop over each face found
    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        # recognize
        matches = face_recognition.compare_faces(data["encodings"], enc, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            best = np.argmin(face_recognition.face_distance(data["encodings"], enc))
            name = data["names"][best]

        # draw tight green box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show result
    cv2.imshow("Face Detection + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
