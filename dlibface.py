import dlib
import cv2
import numpy as np
from scipy.spatial import distance
import time

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load dlib's pre-trained models
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Dictionary to store registered faces' embeddings
face_database = {}


def get_face_embedding_from_frame(frame):
    """
    Detects faces in a given frame and returns a list of (embedding, face_rect)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    results = []
    for face in faces:
        landmarks = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(frame, landmarks)
        embedding = np.array(face_descriptor)
        results.append((embedding, face))

    return results


def get_face_embedding(frames_count=50):
    """
    Captures frames from webcam and returns a stable median face embedding.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return None

    time.sleep(2)
    print(f"📸 Capturing {frames_count} frames. Please move your face slightly...")

    embeddings_list = []
    start_time = time.time()

    for _ in range(frames_count):
        ret, frame = cap.read()
        if not ret:
            continue

        results = get_face_embedding_from_frame(frame)
        if results:
            embedding, _ = results[0]  # Take the first detected face
            embeddings_list.append(embedding)

        # Show preview
        cv2.imshow("Capturing Face - Move Slightly", frame)
        cv2.waitKey(1)

        if time.time() - start_time > 6:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings_list:
        print("❌ No face detected in the frames")
        return None

    median_embedding = np.median(np.array(embeddings_list), axis=0)
    return median_embedding


def register_face(name):
    """
    Registers a face by capturing multiple frames and computing a stable embedding.
    """
    stable_embedding = get_face_embedding(frames_count=50)
    if stable_embedding is not None:
        face_database[name] = [stable_embedding]
        print(f"✅ Successfully registered {name}!")
    else:
        print("❌ No face detected during registration")


def recognize_face():
    """
    Recognizes a person in real-time and draws bounding boxes around faces.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("📸 Recognizing face in real-time... (Press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = get_face_embedding_from_frame(frame)

        if not results:
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for embedding, face in results:
                recognized_name, min_distance = compare_embeddings(embedding)

                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                if min_distance < 0.45:
                    label = f"{recognized_name} ({(1 - min_distance) * 100:.2f}%)"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def compare_embeddings(test_embedding):
    """
    Compares the given face embedding with the database and returns the best match.
    """
    min_distance = float('inf')
    recognized_name = "Unknown"

    for name, known_embeddings in face_database.items():
        for known_embedding in known_embeddings:
            dist = distance.euclidean(test_embedding, known_embedding)
            if dist < min_distance:
                min_distance = dist
                recognized_name = name

    return recognized_name, min_distance


def main():
    """
    Interactive CLI for face registration and real-time recognition.
    """
    while True:
        print("\n1. Register Face")
        print("2. Recognize Face in Real-Time")
        print("3. Exit")
        choice = input("Select option: ")

        if choice == "1":
            name = input("Enter name: ").strip()
            if name:
                register_face(name)
        elif choice == "2":
            recognize_face()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
