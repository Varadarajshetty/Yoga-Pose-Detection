import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# File paths
DATASET_CSV = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\src\pose_keypoints_dataset.csv"
MODEL_PATH = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\src\yoga_pose_classifier.h5"

# Load data from CSV
df = pd.read_csv(DATASET_CSV)

# Separate features and labels
labels = df['label'].values
X = df.drop('label', axis=1).values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

print("X shape:", X.shape)
print("y shape:", y_encoded.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Load trained model
model = load_model(MODEL_PATH)

# One-hot encode test labels for evaluation
num_classes = model.output_shape[1]  # Number of classes, e.g. 107
y_test_cat = to_categorical(y_test, num_classes)

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=2)
print(f"Test accuracy: {accuracy:.4f}")

# Predict classes on test data
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Convert encoded labels back to original string labels
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred_classes)

print("Sample predictions:")
for i in range(5):
    print(f"True: {y_test_labels[i]} - Predicted: {y_pred_labels[i]}")
