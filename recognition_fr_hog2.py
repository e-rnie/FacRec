#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

# ─────────────────────────────────────────────────────────────────────────────
# Load your known face encodings (created by train_encodings.py)
# ─────────────────────────────────────────────────────────────────────────────
data = pickle.load(open("encodings.pickle", "rb"))
# data["encodings"] is a list of 128-D face encodings
# data["names"]     is a list of corresponding person names

# ─────────────────────────────────────────────────────────────────────────────
# Initialize the HOG + SVM person detector (no cv2.dnn needed)
# ─────────────────────────────────────────────────────────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ─────────────────────────────────────────────────────────────────────────────
# Open the camera at low resolution for speed
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

frame_count   = 0
person_boxes  = []

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run the expensive person detector every 5 frames
    if frame_count % 5 == 0:
        # HOG person detector (fast settings)
        rects, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.2
        )
        # convert to [x1, y1, x2, y2] and apply non-max suppression
        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
        person_boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    # For each detected person, detect & recognize faces
    for (startX, startY, endX, endY) in person_boxes:
        # draw person box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # crop the person region of interest (ROI)
        roi = frame[startY:endY, startX:endX]
        # convert BGR -> RGB for face_recognition
        rgb_roi = roi[:, :, ::-1]

        # detect face locations in the ROI
        face_locs = face_recognition.face_locations(rgb_roi, model="hog")
        # compute face encodings for each face
        face_encs = face_recognition.face_encodings(
            rgb_roi,
            known_face_locations=face_locs,
            num_jitters=1
        )

        # loop over each detected face
        for ((top, right, bottom, left), enc) in zip(face_locs, face_encs):
            # compute coordinates relative to the original frame
            t = startY + top
            b = startY + bottom
            l = startX + left
            r = startX + right

            # perform recognition
            matches = face_recognition.compare_faces(
                data["encodings"], enc, tolerance=0.5
            )
            name = "Unknown"
            if True in matches:
                # pick the known face with the smallest distance to the new face
                best_idx = np.argmin(
                    face_recognition.face_distance(data["encodings"], enc)
                )
                name = data["names"][best_idx]

            # draw the face bounding box and label
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, name, (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # display the result
    cv2.imshow("Fast Person & Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
