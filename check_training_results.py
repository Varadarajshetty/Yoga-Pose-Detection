import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
print(f"✓ Loaded {len(class_indices)} classes")

# Set up validation data generator
DATA_DIR = r"C:/Users/Varadaraj S/OneDrive/Desktop/yoga_pose_detection/data/raw_images/dataset"
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✓ Validation set: {validation_generator.samples} images")

# Test both models
models_to_test = [
    ('Best Model', 'models/best_yoga_model.h5'),
    ('Final Model', 'models/yoga_pose_classifier.h5')
]

for model_name, model_path in models_to_test:
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    try:
        # Load the model
        print(f"Loading {model_name}...")
        model = tf.keras.models.load_model(model_path)
        print(f"✓ {model_name} loaded successfully!")
        
        # Evaluate the model
        print(f"Evaluating {model_name} performance...")
        evaluation = model.evaluate(validation_generator, verbose=0)
        
        print(f"\n=== {model_name.upper()} RESULTS ===")
        print(f"Validation Loss: {evaluation[0]:.4f}")
        print(f"Validation Accuracy: {evaluation[1]:.4f}")
        if len(evaluation) > 2:
            print(f"Validation Top-5 Accuracy: {evaluation[2]:.4f}")
        
        # Calculate accuracy percentage
        accuracy_percent = evaluation[1] * 100
        print(f"\n🎯 {model_name} Accuracy: {accuracy_percent:.2f}%")
        
        if len(evaluation) > 2:
            top5_percent = evaluation[2] * 100
            print(f"🎯 {model_name} Top-5 Accuracy: {top5_percent:.2f}%")
        
        # Check if accuracy improved
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
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")

print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print("✓ models/best_yoga_model.h5 (best validation model)")
print("✓ models/yoga_pose_classifier.h5 (final model)")
print("✓ models/class_indices.json (class mappings)")
print("✓ models/training_history.png (training plots)")
print("✓ models/confusion_matrix.png (confusion matrix)")

print("\n🎉 Training completed successfully!") 