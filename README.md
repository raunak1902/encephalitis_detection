# 🧠 Encephalitis Detection AI System

A full-stack AI healthcare application that detects **encephalitis risk** using:

• Brain MRI images (Deep Learning – DenseNet121)  
• Clinical symptoms (Rule-based model)  
• Multimodal fusion (Image + Symptoms)

Built using **React + Flask + PyTorch**

---

## 🚀 Features

• MRI brain scan classification (11 neurological conditions)  
• Clinical symptom risk scoring  
• Multimodal prediction fusion (70% MRI + 30% symptoms)  
• REST API backend (Flask)  
• React frontend UI  
• End-to-end AI pipeline

---

## 🧠 Model Details

Model: DenseNet121 (Transfer Learning)  
Dataset: NINS Brain MRI Dataset  
Classes: 11 brain conditions

Prediction Output:
• Normal  
• Encephalitis Risk

---

## 📂 Project Structure

encephalitis_detection/
│
├── backend/ # Flask API
├── frontend/ # React App
├── training/ # Training scripts (dataset not included)
└── README.md


---

## 📥 Dataset Download

Dataset is too large for GitHub.

Download from:
https://figshare.com/articles/dataset/NINS_Brain_MRI/28399209

After download, place here:

training/NINS_Dataset/


---

## 📥 Download Trained Model

Download model weights from (upload to Google Drive):
densenet_mri_model.pth


Place it here:

training/densenet_mri_model.pth


---

## ⚙️ Run Backend

cd backend
python -m pip install -r requirements.txt
python app.py


Backend runs on:
http://127.0.0.1:8000

---

## ⚙️ Run Frontend

cd frontend
npm install
npm start


Frontend runs on:
http://localhost:3000

---

## 🧪 Usage

Upload MRI image and/or enter clinical symptoms → Get prediction.

---

## 🏆 Tech Stack

Frontend: React  
Backend: Flask  
ML: PyTorch, TorchVision  
Image Processing: Pillow  
Deployment Ready API

---

## 👨‍💻 Author

Final Year AI Project