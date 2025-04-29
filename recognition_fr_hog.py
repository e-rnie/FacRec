import cv2
import face_recognition
import pickle
import imutils
import numpy as np

# --- Load your face encodings ---
data = pickle.load(open("encodings.pickle", "rb"))
# data["encodings"], data["names"]

# --- Initialize HOG pedestrian detector ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- Initialize video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed
    frame = imutils.resize(frame, width=600)
    orig = frame.copy()

    # --- Detect people with HOG ---
    # returns bounding boxes and weights
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05
    )

    # Optional: apply non-max suppression to reduce overlaps
    from imutils.object_detection import non_max_suppression
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    # Loop over the final detected person boxes
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (255, 0, 0), 2)
        cv2.putText(frame, "Person", (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- Within each person ROI, detect & recognize faces ---
        roi = orig[startY:endY, startX:endX]
        # face_recognition wants RGB
        rgb_roi = roi[:, :, ::-1]

        # Detect faces
        locs = face_recognition.face_locations(rgb_roi, model="hog")
        encs = face_recognition.face_encodings(rgb_roi, locs)

        for ((top, right, bottom, left), enc) in zip(locs, encs):
            # scale back face coords to original frame
            top   += startY
            bottom+= startY
            left  += startX
            right += startX

            # Recognition
            matches = face_recognition.compare_faces(data["encodings"], enc, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                best = np.argmin(face_recognition.face_distance(data["encodings"], enc))
                name = data["names"][best]

            # Draw face box + name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("HOG + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
