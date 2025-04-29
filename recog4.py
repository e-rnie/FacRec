#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import numpy as np

# ————————————————
# PARAMETERS
# ————————————————
ENCODINGS_FILE = "encodings.pickle"
CAM_WIDTH      = 320
CAM_HEIGHT     = 240
TOLERANCE      = 0.5     # lower = stricter match
MODEL          = "hog"   # or "cnn" if you installed the CNN detector

# ————————————————
# LOAD YOUR TRAINED ENCODINGS
# ————————————————
data = pickle.load(open(ENCODINGS_FILE, "rb"))
print(f"✅ Loaded {len(data['encodings'])} face encodings.")

# ————————————————
# OPEN CAMERA
# ————————————————
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print(f"📷 Camera open at {CAM_WIDTH}×{CAM_HEIGHT}. Press Q to quit.")

# ————————————————
# MAIN LOOP
# ————————————————
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert BGR to RGB
    rgb = frame[:, :, ::-1]

    # 1) Detect faces (returns list of (top, right, bottom, left))
    face_locs = face_recognition.face_locations(rgb, model=MODEL)

    # Debug
    print(f"Detected {len(face_locs)} face(s)")

    # 2) Only if we have faces do we compute encodings
    if face_locs:
        face_encs = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locs,  # MUST be keyword
            num_jitters=1
        )

        # Debug
        print(f"Computed {len(face_encs)} encodings")

        # 3) If lengths match, label them
        if len(face_encs) == len(face_locs):
            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                matches = face_recognition.compare_faces(data["encodings"], enc, TOLERANCE)
                name = "Unknown"
                if True in matches:
                    best_idx = np.argmin(face_recognition.face_distance(data["encodings"], enc))
                    name = data["names"][best_idx]

                # Draw box + label
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            print("⚠️ face_locs/face_encs length mismatch – skipping labeling")
    else:
        # no faces found
        pass

    # 4) Show the frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
