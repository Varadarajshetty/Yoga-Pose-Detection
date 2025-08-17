import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

DATA_DIR = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/data/raw_images/dataset"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Get a batch
images, labels = next(train_generator)

plt.figure(figsize=(16, 8))
for i in range(min(BATCH_SIZE, 12)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i])
    label_idx = np.argmax(labels[i])
    label_name = inv_class_indices[label_idx]
    plt.title(label_name, fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show() 