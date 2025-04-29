#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import numpy as np

# —————————————
# CONFIGURATION
# —————————————
ENC_FILE   = "encodings.pickle"
CAM_WIDTH  = 320
CAM_HEIGHT = 240
TOLERANCE  = 0.5
MODEL      = "hog"   # or "cnn"

# —————————————
# LOAD KNOWN FACES
# —————————————
data = pickle.load(open(ENC_FILE, "rb"))
print(f"[i] Loaded {len(data['encodings'])} encodings for {len(data['names'])} people")

# —————————————
# OPEN CAMERA
# —————————————
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print(f"[i] Camera streaming at {CAM_WIDTH}×{CAM_HEIGHT}. Press Q to quit.\n")

# —————————————
# MAIN LOOP
# —————————————
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # BGR → RGB
    rgb = frame[:, :, ::-1]

    # 1) detect face_locations (tuples)
    face_locs = face_recognition.face_locations(rgb, model=MODEL)
    print(" face_locs:", face_locs)
    print(" types:    ", [type(x) for x in face_locs])

    # 2) only if we actually found some tuples do we encode
    encodings = []
    if face_locs:
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locs,  # <— must use keyword
            num_jitters=1
        )
        print(" face_encs count:", len(encodings))
    else:
        print(" no faces this frame, skipping face_encodings")

    # 3) if counts line up, draw + recognize
    if len(encodings) == len(face_locs) and face_locs:
        for (top, right, bottom, left), enc in zip(face_locs, encodings):
            matches = face_recognition.compare_faces(data["encodings"], enc, TOLERANCE)
            name = "Unknown"
            if True in matches:
                best = np.argmin(face_recognition.face_distance(data["encodings"], enc))
                name = data["names"][best]
            # draw box + label
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        if face_locs:
            print(" ⚠️  Mismatch locs vs encs, skipping draw")

    # 4) display
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
