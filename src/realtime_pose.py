import cv2
import numpy as np
import tensorflow as tf
import json

# Load model and class labels
MODEL_PATH = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/models/yoga_pose_classifier.h5"
CLASS_INDICES_PATH = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/models/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_pose(frame):
    processed = preprocess_frame(frame)
    predictions = model.predict(processed)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    return labels[class_id], confidence

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        pose, confidence = predict_pose(frame)

        # Put text on frame
        text = f"{pose} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Yoga Pose Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
