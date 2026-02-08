# Backend API for Encephalitis Detection

This directory contains the backend API server for the Encephalitis Detection system.

## 📁 Files

- **app.py** - Flask-based API server (recommended for beginners)
- **app_fastapi.py** - FastAPI-based server (alternative, more modern)
- **requirements.txt** - Python dependencies

## 🚀 Quick Start

### Option 1: Flask (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server will start at: `http://localhost:8000`

### Option 2: FastAPI (Alternative)

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart

# Run server
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

Access API docs at: `http://localhost:8000/docs`

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Predict
```bash
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (file, optional): MRI brain scan image
- `Oxygen_Level` (float, optional): 0-100
- `Body_Temperature` (float, optional): °F
- `Glucose_Level` (float, optional): mg/dL
- `Blood_Pressure` (float, optional): mmHg
- `Fever` (int, optional): 0 or 1

**Response:**
```json
{
  "prediction": "Normal",
  "confidence": 0.85,
  "from": {
    "image": true,
    "symptoms": true
  },
  "details": {
    "image_prediction": "Normal",
    "image_confidence": 0.92,
    "symptom_risk_score": 15,
    "symptom_details": {
      "oxygen_normal": true,
      "temp_normal": true,
      "glucose_normal": true,
      "bp_normal": true,
      "fever_present": false
    }
  }
}
```

### 3. Get Classes
```bash
GET /classes
```

**Response:**
```json
{
  "classes": ["Normal", "Encephalitis", ...],
  "total_classes": 17
}
```

## 🧪 Testing

### Using cURL

**Test image upload:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/mri_scan.jpg"
```

**Test symptoms only:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "Oxygen_Level=95" \
  -F "Body_Temperature=98.6" \
  -F "Glucose_Level=100" \
  -F "Blood_Pressure=120" \
  -F "Fever=0"
```

**Test both:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@mri_scan.jpg" \
  -F "Oxygen_Level=95" \
  -F "Body_Temperature=98.6" \
  -F "Glucose_Level=100" \
  -F "Blood_Pressure=120" \
  -F "Fever=0"
```

### Using Python

```python
import requests

# Test with image
files = {'file': open('mri_scan.jpg', 'rb')}
data = {
    'Oxygen_Level': 95,
    'Body_Temperature': 98.6,
    'Glucose_Level': 100,
    'Blood_Pressure': 120,
    'Fever': 0
}

response = requests.post('http://localhost:8000/predict', 
                        files=files, data=data)
print(response.json())
```

## 🔧 Configuration

### Change Model Path
Edit the `MODEL_PATH` variable in `app.py` or `app_fastapi.py`:

```python
MODEL_PATH = '/path/to/your/model.pth'
```

### Force CPU Usage
If you don't have a GPU or encounter CUDA errors:

```python
DEVICE = torch.device("cpu")
```

### Change Port
```python
# Flask
app.run(host='0.0.0.0', port=5000)

# FastAPI
uvicorn.run(app, host="0.0.0.0", port=5000)
```

## 🏥 Prediction Logic

### Image Prediction
1. Loads MRI image
2. Preprocesses (resize to 224x224, normalize)
3. Runs through DenseNet121 model
4. Returns predicted class + confidence

### Symptom Prediction
Simple rule-based scoring:
- Low oxygen (<90%): +30 risk points
- High fever (>100.4°F): +25 risk points
- Abnormal glucose: +20 risk points
- Abnormal blood pressure: +15 risk points

**Risk Score Interpretation:**
- < 30: Normal (low risk)
- 30-60: Moderate risk
- > 60: High risk (probable encephalitis)

### Combined Prediction
When both image and symptoms are provided:
- **70% weight** on image prediction
- **30% weight** on symptom analysis
- Final prediction is weighted average

## 🐛 Troubleshooting

### Model Not Loading
```
Error: Model file not found
```

**Solution:** Make sure `densenet121_pretrained_mri.pth` exists at `MODEL_PATH`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:** Force CPU usage:
```python
DEVICE = torch.device("cpu")
```

### Module Not Found
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Port Already in Use
```
OSError: [Errno 98] Address already in use
```

**Solution:** Kill existing process or change port:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python app.py --port 5000
```

## 📈 Improving Predictions

### 1. Better Symptom Model
Replace rule-based scoring with ML model:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, 'symptom_model.pkl')

# Use in prediction
clf = joblib.load('symptom_model.pkl')
prediction = clf.predict([symptoms])
```

### 2. Ensemble Multiple Models
Combine predictions from multiple architectures:

```python
# Load multiple models
densenet = load_model('densenet.pth')
resnet = load_model('resnet.pth')
efficientnet = load_model('efficientnet.pth')

# Average predictions
final = (densenet + resnet + efficientnet) / 3
```

### 3. Add Uncertainty Estimation
Use Monte Carlo Dropout for uncertainty:

```python
def predict_with_uncertainty(model, image, n_iter=10):
    model.train()  # Enable dropout
    predictions = []
    
    for _ in range(n_iter):
        pred = model(image)
        predictions.append(pred)
    
    mean = torch.mean(predictions)
    std = torch.std(predictions)
    return mean, std
```

## 🚢 Production Deployment

### Using Gunicorn (Flask)
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Uvicorn (FastAPI)
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
Create `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t encephalitis-api .
docker run -p 8000:8000 encephalitis-api
```

## 📊 Performance Monitoring

Add logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Received prediction request")
    # ... prediction logic
    logger.info(f"Prediction: {result}")
```

## 🔐 Security Considerations

1. **API Rate Limiting**
2. **Input Validation**
3. **Authentication** (add API keys)
4. **HTTPS** in production
5. **File Size Limits**

## 📝 License

Educational/Research Use Only

## 🆘 Support

For issues, check:
1. Server logs
2. Model file exists
3. All dependencies installed
4. Correct Python version (3.8+)
