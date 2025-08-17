# save_label_encoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Path to your dataset
DATASET_CSV = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\src\pose_keypoints_dataset.csv"

# Load labels
df = pd.read_csv(DATASET_CSV)
labels = df['label'].values

# Fit and save the encoder
le = LabelEncoder()
le.fit(labels)

joblib.dump(le, "label_encoder.pkl")
print("label_encoder.pkl saved successfully.")
