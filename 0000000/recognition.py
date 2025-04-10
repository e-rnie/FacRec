# recognition.py
import cv2
import numpy as np
import imutils
import pickle

# -------------------------
# Object Detection Setup (MobileNet SSD)
# -------------------------
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define the class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# -------------------------
# Face Recognition Setup (LBPH)
# -------------------------
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
lbph_recognizer.read("lbph_model.xml")

with open("labelmap_lbph.pkl", "rb") as f:
    lbph_label_map = pickle.load(f)

# -------------------------
# Face Detection Setup (Haar Cascade)
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Video Capture
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare the image for MobileNet SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # -------------------------
                # Face Detection & Recognition within the person bounding box
                # -------------------------
                person_roi = frame[startY:endY, startX:endX]
                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(person_roi, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    face_crop = gray_roi[fy:fy+fh, fx:fx+fw]
                    label_id, rec_confidence = lbph_recognizer.predict(face_crop)
                    text = "Unknown"
                    if rec_confidence < 70:
                        text = lbph_label_map.get(label_id, "Unknown")
                    cv2.putText(person_roi, text, (fx, fy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object & Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
