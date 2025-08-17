import sys
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib  # or use pickle if needed

# Load model and label encoder
MODEL_PATH = r"yoga_pose_classifier.h5"
ENCODER_PATH = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\src\label_encoder.pkl"  # Save this separately during training

model = load_model(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected.")

    keypoints = []
    for kp in results.pose_landmarks.landmark:
        keypoints.extend([kp.x, kp.y, kp.z, kp.visibility])

    keypoints = np.array(keypoints)

    return np.array(keypoints)

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_from_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        keypoints = extract_keypoints(image_path)
        if keypoints.shape[0] != 132:
            print("Unexpected keypoints shape. Got:", keypoints.shape)
            sys.exit(1)

        keypoints = keypoints.reshape(1, -1)  # Shape: (1, 132)
        prediction = model.predict(keypoints)
        pred_class = np.argmax(prediction, axis=1)
        pred_label = le.inverse_transform(pred_class)[0]

        print(f"Predicted Pose: {pred_label}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
