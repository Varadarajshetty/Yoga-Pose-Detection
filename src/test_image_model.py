import tensorflow as tf
import numpy as np
import cv2
import json
import os
import sys

# Load the saved model
MODEL_PATH = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/models/yoga_pose_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices (ensure this file was saved during training)
with open(r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/models/class_indices.json", 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_pose(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    return labels[class_id], confidence


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # default image path if no argument given
        test_image_path = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\data\raw_images\dataset\adho mukha svanasana\1. 1.png"

    pose, confidence = predict_pose(test_image_path)
    print(f"Predicted Pose: {pose} with confidence {confidence:.2f}")

 # Show image with prediction text
    img = cv2.imread(test_image_path)
    cv2.putText(img, f"{pose} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Yoga Pose Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()