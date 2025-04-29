#!/usr/bin/env python3
import cv2, face_recognition, pickle, numpy as np, types

# -----------------------------------------------------------------
print("DEBUG – If an exception is raised, read the text it prints.\n")
data = pickle.load(open("encodings.pickle", "rb"))

cap = cv2.VideoCapture(0)
cap.set(3, 320); cap.set(4, 240)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    rgb = frame[:, :, ::-1]

    # Step-1  detect faces
    face_locs = face_recognition.face_locations(rgb, model="hog")
    # VERIFY every element is a tuple of length 4
    if not all(isinstance(t, tuple) and len(t) == 4 for t in face_locs):
        raise RuntimeError(
            f"\n❌ BAD face_locs!  Expected list of tuples, got:\n{face_locs}\n"
            "Some other detector must be injecting dlib objects."
        )

    # Step-2  only call encodings if we really have tuples
    if face_locs:
        encs = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locs,    # <— keyword
            num_jitters=0
        )
        print(f"OK frame → {len(encs)} encodings")

    # Draw simple boxes for visual feedback
    for (t,r,b,l) in face_locs:
        cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 2)
    cv2.imshow("debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
