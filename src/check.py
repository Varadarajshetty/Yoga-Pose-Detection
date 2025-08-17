import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv('pose_keypoints_dataset.csv')
X = df.values  # Converts dataframe to numpy array
y = np.load('y_labels.npy')     # labels

print("X shape:", X.shape)
print("y shape:", y.shape)

# If shapes differ, handle accordingly here
assert X.shape[0] == y.shape[0], "Mismatch in samples count between X and y!"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
