# recognition_fr.py
import cv2, face_recognition, pickle, imutils, numpy as np

# load encodings
data = pickle.load(open("encodings.pickle","rb"))

# load MobileNet SSD as before
net = cv2.dnn.readNetFromCaffe(
  "models/MobileNetSSD_deploy.prototxt",
  "models/MobileNetSSD_deploy.caffemodel"
)
CLASSES = [ … ]  # same as before

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    (h,w) = frame.shape[:2]

    # object detection
    blob = cv2.dnn.blobFromImage(frame,0.007843,(300,300),127.5)
    net.setInput(blob)
    dets = net.forward()

    for i in range(dets.shape[2]):
        conf = dets[0,0,i,2]
        if conf>0.5 and int(dets[0,0,i,1])==15:  # 15=“person”
            box = (dets[0,0,i,3:7]*np.array([w,h,w,h])).astype("int")
            (sx,sy,ex,ey) = box
            cv2.rectangle(frame,(sx,sy),(ex,ey),(255,0,0),2)

            # face detection & recognition with face_recognition
            rgb = frame[sy:ey, sx:ex][:,:,::-1]
            locations = face_recognition.face_locations(rgb, model="hog")
            encs = face_recognition.face_encodings(rgb, locations)
            for (top,right,bottom,left), enc in zip(locations, encs):
                matches = face_recognition.compare_faces(
                    data["encodings"], enc, tolerance=0.5
                )
                name = "Unknown"
                if True in matches:
                    idx = matches.index(True)
                    name = data["names"][idx]
                # draw on original frame (adjust coords)
                cv2.rectangle(frame, (sx+left, sy+top),
                              (sx+right, sy+bottom),(0,255,0),2)
                cv2.putText(frame, name, (sx+left, sy+top-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.imshow("FR", frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
