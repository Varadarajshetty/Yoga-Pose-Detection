import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# === Step 1: Set dataset path ===
DATA_DIR = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/data/raw_images/dataset"

# === Step 2: Parameters ===
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20

# === Step 3: Simple Data augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Validation data with only rescaling
val_datagen = ImageDataGenerator(
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

# Save class indices to a JSON file
class_indices = train_generator.class_indices
with open('models/class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Saved class indices to models/class_indices.json")

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print("Class indices:", train_generator.class_indices)

# === Step 4: Build simple model ===
def create_simple_model():
    # Use MobileNetV2 (smaller, faster, and was working before)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

model, base_model = create_simple_model()

# Compile with simple settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === Step 5: Simple callbacks ===
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'models/best_yoga_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# === Step 6: Train the model (Single phase) ===
print("Training the model...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# === Step 7: Save the final model ===
model.save('models/yoga_pose_classifier.h5')
print("Final model saved to models/yoga_pose_classifier.h5")

# === Step 8: Plot training history ===
def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# === Step 9: Evaluate model performance ===
print("\n=== Model Evaluation ===")
evaluation = model.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {evaluation[0]:.4f}")
print(f"Validation Accuracy: {evaluation[1]:.4f}")

# Calculate accuracy percentage
accuracy_percent = evaluation[1] * 100
print(f"\n🎯 Final Accuracy: {accuracy_percent:.2f}%")

if accuracy_percent >= 80:
    print("🚀 EXCELLENT! Accuracy is 80% or higher!")
elif accuracy_percent >= 70:
    print("✅ GOOD! Accuracy is 70% or higher!")
elif accuracy_percent >= 60:
    print("📈 IMPROVED! Accuracy is 60% or higher!")
elif accuracy_percent >= 50:
    print("📊 DECENT! Accuracy is 50% or higher!")
else:
    print("⚠️  Accuracy needs further improvement")

print("\nTraining completed successfully!") 