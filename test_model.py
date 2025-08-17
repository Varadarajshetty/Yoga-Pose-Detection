import tensorflow as tf
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load the best model
print("Loading model...")
model = load_model('models/best_yoga_model.h5')
print("✓ Model loaded successfully!")

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}
print(f"✓ Loaded {len(class_names)} classes")

# Test with a sample image from your dataset
test_image_path = "data/raw_images/dataset/tadasana/1.jpg"  # Adjust path as needed

try:
    # Load and preprocess image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"❌ Could not load image: {test_image_path}")
        exit(1)
    
    # Resize and normalize
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image, verbose=0)
    
    # Get top 5 predictions
    top_5_indices = np.argsort(prediction[0])[-5:][::-1]
    top_5_confidences = prediction[0][top_5_indices]
    
    print("\n🎯 Model Test Results:")
    print("=" * 50)
    
    for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
        pose_name = class_names.get(idx, "Unknown")
        print(f"{i+1}. {pose_name.title()}: {conf:.3f} ({conf*100:.1f}%)")
    
    # Check if top prediction is reasonable
    top_prediction = class_names.get(top_5_indices[0], "Unknown")
    top_confidence = top_5_confidences[0]
    
    print(f"\n✅ Top Prediction: {top_prediction.title()}")
    print(f"✅ Confidence: {top_confidence:.3f} ({top_confidence*100:.1f}%)")
    
    if top_confidence > 0.1:  # Very low threshold for testing
        print("🎉 Model is working correctly!")
    else:
        print("⚠️  Model confidence is very low - may need investigation")
        
except Exception as e:
    print(f"❌ Error testing model: {str(e)}") 