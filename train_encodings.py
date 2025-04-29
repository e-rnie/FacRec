# train_encodings.py
import os, pickle, face_recognition

known_encodings, known_names = [], []

for person in os.listdir("known_faces"):
    person_dir = os.path.join("known_faces", person)
    if not os.path.isdir(person_dir): continue
    for img in os.listdir(person_dir):
        if img.lower().endswith(('.jpg','png','jpeg')):
            image = face_recognition.load_image_file(os.path.join(person_dir, img))
            encs = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(person)

with open("encodings.pickle","wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)
print("Saved encodings.pickle")
