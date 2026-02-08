"""
Encephalitis Detection Backend API (FastAPI Version)
FastAPI-based REST API for MRI image and clinical symptom analysis

This is an alternative to the Flask version with better async support
and automatic API documentation.

To use this instead of app.py:
    pip install fastapi uvicorn python-multipart
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel

# Configuration
MODEL_PATH = r"C:\Users\rauna\Downloads\encephalitis_detection\training\densenet_mri_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = [
    'Brain Atrophy',
    'Brain Infection',
    'Brain Infection with abscess',
    'Cerebral Hemorrhage',
    'Cerebral abscess',
    'Demyelinating lesions',
    'Encephalomalacia with gliotic change',
    'Focal pachymeningitis',
    'Ischemic change demyelinating plaque',
    'Leukoencephalopathy with subcortical cysts',
    'Microvascular ischemic change',
    'NMOSD ADEM',
    'Normal',
    'Obstructive Hydrocephalus',
    'Postoperative encephalomalacia',
    'Stroke (Demyelination)',
    'White Matter Disease'
]

# Image preprocessing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    from_sources: dict
    details: dict

# Initialize FastAPI app
app = FastAPI(
    title="Encephalitis Detection API",
    description="AI-powered MRI analysis and clinical symptom evaluation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EncephalitisModel:
    """Model wrapper"""
    
    def __init__(self):
        self.model = None
        self.device = DEVICE
        
    def load_model(self):
        """Load DenseNet121 model"""
        print(f"Loading model from {MODEL_PATH}...")
        print(f"Using device: {self.device}")
        
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_NAMES))
        )
        
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_image(self, image_bytes):
        """Predict from image"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
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


def analyze_symptoms(oxygen: float, temp: float, glucose: float, 
                     bp: float, fever: int):
    """Analyze clinical symptoms"""
    risk_score = 0
    
    if oxygen < 90:
        risk_score += 30
    elif oxygen < 95:
        risk_score += 15
        
    if temp > 100.4 or fever == 1:
        risk_score += 25
    elif temp > 99:
        risk_score += 10
        
    if glucose < 70 or glucose > 180:
        risk_score += 20
    elif glucose < 80 or glucose > 140:
        risk_score += 10
        
    if bp < 90 or bp > 140:
        risk_score += 15
    elif bp < 100 or bp > 130:
        risk_score += 5
        
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


# Initialize model
encephalitis_model = EncephalitisModel()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        encephalitis_model.load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Encephalitis Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": encephalitis_model.model is not None,
        "device": str(DEVICE)
    }


@app.post("/predict", tags=["Prediction"])
async def predict(
    file: Optional[UploadFile] = File(None),
    Oxygen_Level: Optional[float] = Form(None),
    Body_Temperature: Optional[float] = Form(None),
    Glucose_Level: Optional[float] = Form(None),
    Blood_Pressure: Optional[float] = Form(None),
    Fever: Optional[int] = Form(0)
):
    """
    Predict encephalitis from MRI image and/or clinical symptoms
    
    - **file**: MRI brain scan image
    - **Oxygen_Level**: Blood oxygen level (%)
    - **Body_Temperature**: Body temperature (°F)
    - **Glucose_Level**: Blood glucose (mg/dL)
    - **Blood_Pressure**: Blood pressure (mmHg)
    - **Fever**: Fever present (0=No, 1=Yes)
    """
    if encephalitis_model.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    has_image = file is not None
    has_symptoms = any(x is not None for x in 
                      [Oxygen_Level, Body_Temperature, Glucose_Level, Blood_Pressure])
    
    if not has_image and not has_symptoms:
        raise HTTPException(
            status_code=400,
            detail="Provide either an MRI image or clinical symptoms"
        )
    
    image_result = None
    symptom_result = None
    
    # Process image
    if has_image:
        contents = await file.read()
        image_result = encephalitis_model.predict_image(contents)
    
    # Process symptoms
    if has_symptoms:
        symptom_result = analyze_symptoms(
            Oxygen_Level or 95,
            Body_Temperature or 98.6,
            Glucose_Level or 100,
            Blood_Pressure or 120,
            Fever or 0
        )
    
    # Combine predictions
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
            'from': {'image': True, 'symptoms': True},
            'details': {
                'image_prediction': image_result['predicted_class'],
                'image_confidence': float(image_result['confidence']),
                'symptom_risk_score': symptom_result['risk_score'],
                'symptom_details': symptom_result['details']
            }
        }
    elif image_result:
        response = {
            'prediction': 'Normal' if image_result['is_normal'] else 'Encephalitis',
            'confidence': float(image_result['confidence']),
            'from': {'image': True, 'symptoms': False},
            'details': {
                'predicted_class': image_result['predicted_class'],
                'all_probabilities': image_result['all_probabilities']
            }
        }
    else:
        response = {
            'prediction': 'Normal' if symptom_result['is_normal'] else 'Encephalitis',
            'confidence': float(symptom_result['confidence']),
            'from': {'image': False, 'symptoms': True},
            'details': {
                'risk_score': symptom_result['risk_score'],
                'symptom_details': symptom_result['details']
            }
        }
    
    return response


@app.get("/classes", tags=["Information"])
async def get_classes():
    """Get all supported brain condition classes"""
    return {
        'classes': CLASS_NAMES,
        'total_classes': len(CLASS_NAMES)
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🧠 Encephalitis Detection API Server (FastAPI)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
