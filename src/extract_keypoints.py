import cv2
import os
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

DATASET_DIR = r"C:\Users\Varadaraj S\OneDrive\Desktop\yoga_pose_detection\data\raw_images\dataset"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

keypoints_data = []
processed_images = []  # <-- New list to save processed image paths

for root, dirs, files in os.walk(DATASET_DIR):
    for file in tqdm(files):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, file)
            class_label = os.path.basename(root)

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                row.append(class_label)
                keypoints_data.append(row)
                processed_images.append(image_path)  # Save full path or filename

# Column names
columns = []
for i in range(33):
    columns.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])
columns.append("label")

# Save keypoints CSV
df = pd.DataFrame(keypoints_data, columns=columns)
df.to_csv("pose_keypoints_dataset.csv", index=False)

# Save processed image paths for label alignment
with open("processed_images.txt", "w") as f:
    for img_path in processed_images:
        f.write(img_path + "\n")

print("✅ Keypoints saved to pose_keypoints_dataset.csv")
print("✅ Processed image paths saved to processed_images.txt")
