"""
Encephalitis Detection Backend API
Flask-based REST API for MRI image and clinical symptom analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
MODEL_PATH = r"C:\Users\rauna\Downloads\encephalitis_detection\training\densenet_mri_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (17 (Now, 11)brain conditions from NINS dataset)
CLASS_NAMES = [
    'Cerebral Hemorrhage',
    'Encephalomalacia with gliotic change',
    'Ischemic change  demyelinating plaque',
    'Microvascular ischemic change',
    'NMOSD  ADEM',
    'Normal',
    'Obstructive Hydrocephalus',
    'White Matter Disease',
    'demyelinating lesions',
    'focal pachymeningitis',
    'Leukoencephalopathy with subcortical cysts'
]


# Image preprocessing pipeline (same as training)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class EncephalitisModel:
    """Wrapper class for the DenseNet model"""
    
    def __init__(self):
        self.model = None
        self.device = DEVICE
        
    def load_model(self):
        """Load the trained DenseNet121 model"""
        print(f"Loading model from {MODEL_PATH}...")
        print(f"Using device: {self.device}")
        
        # Create model architecture
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        
        # Recreate the custom classifier (must match training architecture)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_NAMES))  # 17 classes
        )
        
        # Load trained weights
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
            print("Please train the model first using untitled12.py")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_image(self, image_bytes):
        """
        Predict brain condition from MRI image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            dict: Prediction results with class name and confidence
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = val_transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Determine if it's encephalitis-related
            # Any class except "Normal" is considered encephalitis/brain abnormality
            is_normal = predicted_class == 'Normal'
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'is_normal': is_normal,
                'all_probabilities': {
                    CLASS_NAMES[i]: float(probabilities[0][i])
                    for i in range(len(CLASS_NAMES))
                }
            }
            
        except Exception as e:
            print(f"Error in image prediction: {e}")
            raise


def analyze_symptoms(symptoms):
    """
    Analyze clinical symptoms to predict encephalitis risk
    
    This is a simple rule-based classifier. 
    For production, replace with a trained ML model (Random Forest, XGBoost, etc.)
    
    Args:
        symptoms: dict with Oxygen_Level, Body_Temperature, Glucose_Level, 
                  Blood_Pressure, Fever
                  
    Returns:
        dict: Prediction results
    """
    try:
        # Extract symptom values
        oxygen = float(symptoms.get('Oxygen_Level', 95))
        temp = float(symptoms.get('Body_Temperature', 98.6))
        glucose = float(symptoms.get('Glucose_Level', 100))
        bp = float(symptoms.get('Blood_Pressure', 120))
        fever = int(symptoms.get('Fever', 0))
        
        # Simple risk scoring (0-100)
        risk_score = 0
        
        # Oxygen level (normal: 95-100%)
        if oxygen < 90:
            risk_score += 30
        elif oxygen < 95:
            risk_score += 15
            
        # Body temperature (normal: 97-99°F)
        if temp > 100.4 or fever == 1:
            risk_score += 25
        elif temp > 99:
            risk_score += 10
            
        # Glucose (normal: 70-130 mg/dL)
        if glucose < 70 or glucose > 180:
            risk_score += 20
        elif glucose < 80 or glucose > 140:
            risk_score += 10
            
        # Blood pressure (normal: 90-140 mmHg)
        if bp < 90 or bp > 140:
            risk_score += 15
        elif bp < 100 or bp > 130:
            risk_score += 5
            
        # Determine prediction
        is_normal = risk_score < 30
        confidence = min(risk_score / 100, 0.95) if not is_normal else max(1 - risk_score / 100, 0.60)
        
        return {
            'is_normal': is_normal,
            'confidence': confidence,
            'risk_score': risk_score,
            'details': {
                'oxygen_normal': 95 <= oxygen <= 100,
                'temp_normal': 97 <= temp <= 99,
                'glucose_normal': 70 <= glucose <= 130,
                'bp_normal': 90 <= bp <= 140,
                'fever_present': fever == 1
            }
        }
        
    except Exception as e:
        print(f"Error in symptom analysis: {e}")
        raise


# Initialize model at startup
encephalitis_model = EncephalitisModel()

try:
    encephalitis_model.load_model()
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    print("API will start but predictions will fail until model is available")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': encephalitis_model.model is not None,
        'device': str(DEVICE)
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if encephalitis_model.model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # ----- Detect what user sent -----
        has_image = 'file' in request.files

        symptom_fields = ['Oxygen_Level', 'Body_Temperature', 'Glucose_Level', 'Blood_Pressure']
        has_symptoms = any(request.form.get(field) not in [None, ''] for field in symptom_fields)

        if not has_image and not has_symptoms:
            return jsonify({'error': 'Provide MRI or symptoms'}), 400

        image_result = None
        symptom_result = None

        # ----- IMAGE PREDICTION -----
        if has_image:
            file = request.files['file']
            image_bytes = file.read()
            image_result = encephalitis_model.predict_image(image_bytes)

        # ----- SYMPTOM PREDICTION -----
        if has_symptoms:
            symptoms = {
                'Oxygen_Level': request.form.get('Oxygen_Level') or 95,
                'Body_Temperature': request.form.get('Body_Temperature') or 98.6,
                'Glucose_Level': request.form.get('Glucose_Level') or 100,
                'Blood_Pressure': request.form.get('Blood_Pressure') or 120,
                'Fever': request.form.get('Fever') or 0
            }
            symptom_result = analyze_symptoms(symptoms)

        # ----- COMBINE BOTH -----
        if image_result and symptom_result:
            combined_confidence = (
                (0 if image_result['is_normal'] else image_result['confidence']) * 0.7 +
                (0 if symptom_result['is_normal'] else symptom_result['confidence']) * 0.3
            )

            is_normal = combined_confidence < 0.5
            final_confidence = 1 - combined_confidence if is_normal else combined_confidence

            response = {
                'prediction': 'Normal' if is_normal else 'Encephalitis',
                'confidence': float(final_confidence),
                'from': {'image': True, 'symptoms': True}
            }

        elif image_result:
            response = {
                'prediction': 'Normal' if image_result['is_normal'] else 'Encephalitis',
                'confidence': float(image_result['confidence']),
                'from': {'image': True, 'symptoms': False}
            }

        else:
            response = {
                'prediction': 'Normal' if symptom_result['is_normal'] else 'Encephalitis',
                'confidence': float(symptom_result['confidence']),
                'from': {'image': False, 'symptoms': True}
            }

        return jsonify(response)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500



@app.route('/classes', methods=['GET'])
def get_classes():
    """Return all supported brain condition classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'total_classes': len(CLASS_NAMES)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 Encephalitis Detection API Server")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("="*60 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=8000, debug=True)
