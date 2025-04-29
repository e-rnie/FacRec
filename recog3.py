#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import numpy as np
import traceback

# ————————————————
# CONFIGURATION
# ————————————————
ENCODINGS_FILE = "encodings.pickle"
CAM_WIDTH      = 320
CAM_HEIGHT     = 240
TOLERANCE      = 0.5    # lower = stricter match
MODEL          = "hog"  # or "cnn" if you have GPU & the library installed

# ————————————————
# LOAD KNOWN FACE ENCODINGS
# ————————————————
try:
    data = pickle.load(open(ENCODINGS_FILE, "rb"))
    print(f"✅ Loaded {len(data['encodings'])} encodings for {len(data['names'])} people.")
except Exception:
    print(f"❌ Could not load {ENCODINGS_FILE}. Did you run train_encodings.py?")
    raise

# ————————————————
# OPEN CAMERA
# ————————————————
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print(f"📷 Camera streaming at {CAM_WIDTH}×{CAM_HEIGHT}. Press Q to quit.")

# ————————————————
# MAIN LOOP
# ————————————————
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Empty frame, skipping.")
        continue

    try:
        # 1) Convert BGR to RGB
        rgb = frame[:, :, ::-1]

        # 2) Detect face locations
        face_locs = face_recognition.face_locations(rgb, model=MODEL)

        # 3) Compute encodings for each face location
        face_encs = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locs,  # must be keyword
            num_jitters=1
        )

        # DEBUG: show counts
        print(f"Detected {len(face_locs)} faces, {len(face_encs)} encodings")

        # 4) If lengths mismatch, skip this frame
        if len(face_locs) != len(face_encs):
            print("⚠️  face_locs vs face_encs mismatch → skipping frame")
        else:
            # 5) Loop over each face
            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                # Compare to known encodings
                matches = face_recognition.compare_faces(data["encodings"], enc, TOLERANCE)
                name = "Unknown"
                if True in matches:
                    # Pick the known encoding with smallest distance
                    best_idx = np.argmin(
                        face_recognition.face_distance(data["encodings"], enc)
                    )
                    name = data["names"][best_idx]

                # Draw a tight box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception:
        print("🔥 Exception in recognition loop:")
        traceback.print_exc()

    # 6) Display the frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ————————————————
# CLEANUP
# ————————————————
cap.release()
cv2.destroyAllWindows()
