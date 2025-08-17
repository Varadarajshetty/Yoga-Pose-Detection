from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
import base64
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the trained model
model = load_model('models/best_yoga_model.h5')  # Use the best model instead of final model

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def preprocess_image(image):
    """Enhanced preprocessing for better model performance"""
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Apply additional preprocessing for better accuracy
    # Normalize
    image = image / 255.0
    
    # Apply slight Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def detect_pose(frame):
    """Enhanced pose detection with better error handling"""
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image and find pose
    results = pose.process(image)
    
    # Convert the image back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    prediction_text = "No pose detected"
    confidence = 0.0
    top_predictions = []
    
    if results.pose_landmarks:
        # Draw the pose annotation on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Preprocess for model prediction
        processed_image = preprocess_image(image)
        
        try:
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions for better confidence assessment
            top_5_indices = np.argsort(prediction[0])[-5:][::-1]
            top_5_confidences = prediction[0][top_5_indices]
            
            predicted_class_index = top_5_indices[0]
            predicted_class_name = class_names.get(predicted_class_index, "Unknown Pose")
            confidence = float(top_5_confidences[0])
            
            # Create top predictions list
            top_predictions = []
            for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
                pose_name = class_names.get(idx, "Unknown")
                top_predictions.append({
                    'rank': i + 1,
                    'pose': pose_name,
                    'confidence': float(conf)
                })
            
            # Enhanced confidence threshold and multiple prediction display
            if confidence > 0.3:  # Lower threshold for more predictions
                prediction_text = f"{predicted_class_name.title()} ({confidence:.2f})"
            else:
                # Show top 3 predictions if confidence is low
                top_3_text = []
                for pred in top_predictions[:3]:
                    top_3_text.append(f"{pred['pose'].title()} ({pred['confidence']:.2f})")
                prediction_text = " | ".join(top_3_text)
                
        except Exception as e:
            prediction_text = f"Prediction error: {str(e)}"
    
    # Add text to frame
    cv2.putText(image, prediction_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image, prediction_text, confidence, top_predictions

def analyze_uploaded_image(image_path):
    """Enhanced analysis for uploaded images"""
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            return "Error: Could not read image", 0.0, None, []
        
        # Detect pose
        processed_frame, prediction_text, confidence, top_predictions = detect_pose(frame)
        
        return prediction_text, confidence, processed_frame, top_predictions
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}", 0.0, None, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Enhanced endpoint to receive image data and return prediction"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect pose
        processed_frame, prediction_text, confidence, top_predictions = detect_pose(frame)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': prediction_text,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'processed_image': f'data:image/jpeg;base64,{processed_image_data}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Enhanced endpoint to handle file uploads and return prediction"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '' or file.filename is None:
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please upload an image file.'
            })
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the uploaded image
        result = analyze_uploaded_image(filepath)
        
        # Clean up the uploaded file
        try:
            os.remove(filepath)
        except:
            pass  # Ignore cleanup errors
        
        if len(result) == 4:
            prediction_text, confidence, processed_frame, top_predictions = result
        else:
            prediction_text, confidence = result
            processed_frame = None
            top_predictions = []
        
        if processed_frame is not None:
            # Convert processed frame to base64
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_image_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'prediction': prediction_text,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'processed_image': f'data:image/jpeg;base64,{processed_image_data}'
            })
        else:
            return jsonify({
                'success': False,
                'error': prediction_text
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
