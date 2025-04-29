#!/usr/bin/env python3
import cv2
import face_recognition
import pickle

ENC_FILE = "encodings.pickle"
CAM_WIDTH  = 320
CAM_HEIGHT = 240

# Load your saved encodings
data = pickle.load(open(ENC_FILE, "rb"))
print(f"🔑 Loaded {len(data['encodings'])} known face encodings.")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print(f"📷 Streaming at {CAM_WIDTH}×{CAM_HEIGHT}. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = frame[:, :, ::-1]

    # 1) detect face locations
    face_locs = face_recognition.face_locations(rgb, model="hog")
    print("face_locs:", face_locs)

    # 2) only if we actually got some boxes, compute encodings
    if face_locs:
        face_encs = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locs,
            num_jitters=0
        )
        print(" face_encs count:", len(face_encs))
    else:
        print(" no faces found this frame")

    # draw whatever boxes we got
    for (top, right, bottom, left) in face_locs:
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    cv2.imshow("Debug Face Locs", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
