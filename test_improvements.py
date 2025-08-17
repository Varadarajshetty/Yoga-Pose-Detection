#!/usr/bin/env python3
"""
Test script to verify the improved yoga pose detection system
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import cv2

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        from tensorflow.keras.models import load_model
        model = load_model('models/yoga_pose_classifier.h5')
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_class_indices():
    """Test if class indices are loaded correctly"""
    try:
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        print(f"✅ Class indices loaded: {len(class_indices)} classes")
        return True
    except Exception as e:
        print(f"❌ Class indices loading failed: {e}")
        return False

def test_preprocessing():
    """Test the improved preprocessing function"""
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        from app import preprocess_image
        processed = preprocess_image(dummy_image)
        
        expected_shape = (1, 224, 224, 3)
        if processed.shape == expected_shape:
            print("✅ Preprocessing works correctly")
            return True
        else:
            print(f"❌ Preprocessing shape mismatch: {processed.shape} vs {expected_shape}")
            return False
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe pose detection"""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        print("✅ MediaPipe pose detection initialized")
        return True
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_upload_directory():
    """Test if upload directory exists"""
    if os.path.exists('uploads'):
        print("✅ Upload directory exists")
        return True
    else:
        print("❌ Upload directory missing")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    required_packages = [
        'tensorflow',
        'opencv-python',
        'mediapipe',
        'flask',
        'pillow',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    else:
        print("✅ All required dependencies available")
        return True

def main():
    """Run all tests"""
    print("🧘‍♀️ Testing Yoga Pose Detection System Improvements")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading),
        ("Class Indices", test_class_indices),
        ("Preprocessing", test_preprocessing),
        ("MediaPipe", test_mediapipe),
        ("Upload Directory", test_upload_directory),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for improved accuracy.")
        print("\n📋 Next steps:")
        print("1. Run the improved training: python src/improved_train_model.py")
        print("2. Test the web app: python app.py")
        print("3. Monitor accuracy improvements")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure model files exist in models/ directory")
        print("3. Check file permissions and paths")

if __name__ == "__main__":
    main() 