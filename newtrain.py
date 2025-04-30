import os, cv2, numpy as np, pickle

KNOWN_DIR = "known_faces"
LBPH_OUT  = "lbph_model.xml"
MAP_OUT   = "labelmap.pkl"

images, labels, label_map = [], [], {}
curr_id = 0

for person in os.listdir(KNOWN_DIR):
    p_path = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(p_path): continue
    label_map[curr_id] = person
    for f in os.listdir(p_path):
        if not f.lower().endswith((".jpg",".png",".jpeg")): continue
        img = cv2.imread(os.path.join(p_path, f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        images.append(img); labels.append(curr_id)
    curr_id += 1

print(f"Training on {len(images)} images …")
rec = cv2.face.LBPHFaceRecognizer_create()
rec.train(images, np.array(labels))
rec.save(LBPH_OUT)
pickle.dump(label_map, open(MAP_OUT,"wb"))
print("Saved", LBPH_OUT, MAP_OUT)
