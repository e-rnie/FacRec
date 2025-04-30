#!/usr/bin/env python3
import cv2, numpy as np, pickle, imutils

# — Load models —
SSD_PROTO = "models/MobileNetSSD_deploy.prototxt"
SSD_MODEL = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
           "cat","chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read("lbph_model.xml")
label_map = pickle.load(open("labelmap.pkl","rb"))

# — Camera —
cap = cv2.VideoCapture(0)
cap.set(3,320); cap.set(4,240)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = imutils.resize(frame, width=400)
    (h,w) = frame.shape[:2]

    # 1) Person detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    dets = net.forward()

    for i in range(dets.shape[2]):
        conf = dets[0,0,i,2]
        if conf < 0.5: continue
        idx = int(dets[0,0,i,1])
        if CLASSES[idx] != "person": continue
        box = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
        (x1,y1,x2,y2) = box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        # 2) Face detection inside person ROI
        roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(roi_gray, 1.1, 5)

        for (fx,fy,fw,fh) in faces:
            face_img = roi_gray[fy:fy+fh, fx:fx+fw]
            label_id, dist = lbph.predict(face_img)
            name = "Unknown"
            if dist < 70:
                name = label_map.get(label_id,"Unknown")
            # draw face box + label
            cv2.rectangle(frame,(x1+fx,y1+fy),(x1+fx+fw,y1+fy+fh),(0,255,0),2)
            cv2.putText(frame,name,(x1+fx,y1+fy-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.imshow("OpenCV Person+Face", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
