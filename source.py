import face_recognition
import cv2
import os
import pickle
import subprocess

KNOWN_FACES_DIR = "/Users/krutikakatke/Documents/dataset/faculty/known_faces"
ENCODINGS_FILE = "encodings.pickle"

# ------------------------------------------
# 1. LOAD OR CREATE ENCODINGS
# ------------------------------------------
if os.path.exists(ENCODINGS_FILE):
    print("🔹 Loading saved encodings...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
else:
    print("🔹 No saved encodings found — generating new encodings...")
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            print("Processing:", filename)
            img_path = os.path.join(KNOWN_FACES_DIR, filename)

            image = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(image)

            if len(enc) == 0:
                print(f"⚠️ No face found in {filename} — skipping!")
                continue

            known_face_encodings.append(enc[0])

            # ⭐ ALWAYS USE YOUR REAL NAME ⭐
            name = "Krutika"
            known_face_names.append(name)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

    print("✅ Encodings saved! Next time loading will be instant.")

print("Loaded known faces:", known_face_names)

# ------------------------------------------
# 2. START CAMERA
# ------------------------------------------
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Camera failed to open! Check permissions in macOS Settings → Privacy → Camera")
    exit()

print("🎥 Camera started. Press 'q' to quit.")

# Keep track of who has already been greeted
greeted = set()

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
        matches = face_recognition.compare_faces(known_face_encodings, face_enc)
        name = "Unknown"

        if True in matches:
            name = "Krutika"  # ⭐ ALWAYS SHOW THIS ⭐

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ------------------------------------------
        # 3. VOICE GREETING — spoken once
        # ------------------------------------------
        if name == "Krutika" and name not in greeted:
            greeted.add(name)
            text = "Hello Krutika"
            print("🔊 Saying:", text)
            subprocess.run(["say", text])   # macOS TTS

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()