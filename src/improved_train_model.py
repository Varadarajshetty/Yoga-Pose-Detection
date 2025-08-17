import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
import matplotlib.pyplot as plt
import os

# === Step 1: Set dataset path ===
DATA_DIR = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/data/raw_images/dataset"

# === Step 2: Parameters ===
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16  # Reduced batch size for better generalization
EPOCHS = 5  # Quick test run

# === Step 3: Enhanced Data augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
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

# === Step 4: Build improved model ===
def create_improved_model():
    # Use EfficientNetB0 instead of MobileNetV2 for better performance
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

model, base_model = create_improved_model()

# Compile with better optimizer settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

model.summary()

# === Step 5: Callbacks for better training ===
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when plateau is reached
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
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

# === Step 6: Train the model (Phase 1: Frozen base model) ===
print("Phase 1: Training with frozen base model...")
history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# === Step 7: Fine-tuning (Phase 2: Unfreeze some layers) ===
print("Phase 2: Fine-tuning with unfrozen layers...")

# Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze the bottom layers (keep early layers frozen)
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

# Continue training with fine-tuning
history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,  # Changed from 20 to 50
    callbacks=callbacks,
    verbose=1
)

# === Step 8: Save the final model ===
model.save('models/yoga_pose_classifier.h5')
print("Final model saved to models/yoga_pose_classifier.h5")

# === Step 9: Plot training history ===
def plot_training_history(history1, history2):
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    epochs = range(1, len(combined_history['accuracy']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs, combined_history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, combined_history['val_accuracy'], 'r-', label='Validation Accuracy')
    if 'top_5_accuracy' in combined_history:
        plt.plot(epochs, combined_history['top_5_accuracy'], 'g-', label='Training Top-5 Accuracy')
        plt.plot(epochs, combined_history['val_top_5_accuracy'], 'm-', label='Validation Top-5 Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, combined_history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, combined_history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    if 'lr' in combined_history:
        plt.plot(epochs, combined_history['lr'], 'g-', label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history1, history2)

# === Step 10: Evaluate model performance ===
print("\n=== Model Evaluation ===")
evaluation = model.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {evaluation[0]:.4f}")
print(f"Validation Accuracy: {evaluation[1]:.4f}")
if len(evaluation) > 2:
    print(f"Validation Top-5 Accuracy: {evaluation[2]:.4f}")

# === Step 11: Generate classification report ===
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_classes, target_names=list(class_indices.keys())))

# Plot confusion matrix (sample of classes for readability)
plt.figure(figsize=(20, 16))
cm = confusion_matrix(y_true, y_pred_classes)
# Show only first 20 classes for better visualization
cm_sample = cm[:20, :20]
class_names_sample = list(class_indices.keys())[:20]

sns.heatmap(cm_sample, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names_sample, yticklabels=class_names_sample)
plt.title('Confusion Matrix (First 20 Classes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining completed successfully!")
print("Model files saved:")
print("- models/yoga_pose_classifier.h5 (final model)")
print("- models/best_yoga_model.h5 (best validation model)")
print("- models/class_indices.json (class mappings)")
print("- models/training_history.png (training plots)")
print("- models/confusion_matrix.png (confusion matrix)") 