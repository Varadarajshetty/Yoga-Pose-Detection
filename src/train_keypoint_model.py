import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load data from CSV
df = pd.read_csv("pose_keypoints_dataset.csv")

X = df.drop(columns=["label"]).values  # Features
y_raw = df["label"].values              # Labels from CSV

print("X shape:", X.shape)
print("y shape:", y_raw.shape)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)

y = to_categorical(y_encoded, num_classes=num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded)

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model
model.fit(X_train, y_train,
          epochs=30,
          batch_size=32,
          validation_data=(X_test, y_test))

model.save("yoga_pose_classifier.h5")
print("✅ Model saved as yoga_pose_classifier.h5")
