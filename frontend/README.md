# Encephalitis Detection Frontend

A modern React frontend for the AI-powered encephalitis detection system.

## Features

- 🎨 Modern, responsive UI with glass morphism design
- 📁 Drag & drop file upload
- 🖼️ Image preview before analysis
- ⚡ Real-time AI analysis
- 📱 Mobile-friendly interface
- 🎯 Clear result visualization

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Make sure your FastAPI backend is running on `http://localhost:8000`

## Usage

1. Open the application in your browser
2. Click the upload area or drag & drop an MRI image
3. Preview the selected image
4. Click "Analyze Scan" to get the AI prediction
5. View the results with clear visual indicators

## Technologies Used

- React 18
- Tailwind CSS
- Axios for API calls
- Lucide React for icons
- Glass morphism design

## API Integration

The frontend automatically connects to the FastAPI backend running on port 8000. Make sure your backend is running before using the frontend. 