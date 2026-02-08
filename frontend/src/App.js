import React, { useState } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Brain, 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  Loader, 
  Thermometer, 
  Droplets, 
  Heart, 
  Activity as FeverIcon,
  X,
  FileImage,
  Stethoscope
} from 'lucide-react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [symptoms, setSymptoms] = useState({
    Oxygen_Level: '',
    Body_Temperature: '',
    Glucose_Level: '',
    Blood_Pressure: '',
    Fever: 0
  });

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSymptomChange = (field, value) => {
    setSymptoms(prev => ({
      ...prev,
      [field]: value
    }));
    setPrediction(null);
    setError(null);
  };

  const validateSymptoms = () => {
    const required = ['Oxygen_Level', 'Body_Temperature', 'Glucose_Level', 'Blood_Pressure'];
    for (const field of required) {
      if (!symptoms[field] || symptoms[field] === '') {
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    const formData = new FormData();

    // Add image if selected
    if (selectedFile) {
      formData.append('file', selectedFile);
    }

    /* Add symptoms
    Object.keys(symptoms).forEach(key => {
      formData.append(key, symptoms[key]);
    });

    CHANGING FOR WITHOUT SYMPTOMS i.e, IMAGE ONLY
    */

    // Add symptoms ONLY if user filled them
if (symptoms.Oxygen_Level !== '') {
  formData.append('Oxygen_Level', symptoms.Oxygen_Level);
}
if (symptoms.Body_Temperature !== '') {
  formData.append('Body_Temperature', symptoms.Body_Temperature);
}
if (symptoms.Glucose_Level !== '') {
  formData.append('Glucose_Level', symptoms.Glucose_Level);
}
if (symptoms.Blood_Pressure !== '') {
  formData.append('Blood_Pressure', symptoms.Blood_Pressure);
}

// Fever always has value (0 or 1)
formData.append('Fever', symptoms.Fever);

    // Validate that we have at least some data
    if (!selectedFile && !validateSymptoms()) {
      setError('Please provide an MRI image and/or complete symptom information.');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (err) {
      setError('Error processing request. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    setSymptoms({
      Oxygen_Level: '',
      Body_Temperature: '',
      Glucose_Level: '',
      Blood_Pressure: '',
      Fever: 0
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-red-600';
    if (confidence >= 0.6) return 'text-orange-600';
    return 'text-yellow-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 flex items-center justify-center p-4">
      <div className="max-w-6xl w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-12 h-12 text-white mr-3" />
            <h1 className="text-4xl font-bold text-white">🧠 Encephalitis Detection AI</h1>
          </div>
          <p className="text-white/80 text-lg">
            Comprehensive AI-powered diagnosis using MRI scans and clinical symptoms
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - MRI Upload */}
            <div>
              <h3 className="text-white text-xl font-semibold mb-6 flex items-center">
                <FileImage className="w-6 h-6 mr-2" />
                MRI Brain Scan
              </h3>
              
              <div className="border-2 border-dashed border-white/30 rounded-xl p-6 text-center hover:border-white/50 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 text-white/60 mx-auto mb-4" />
                  <p className="text-white text-lg mb-2">
                    {selectedFile ? selectedFile.name : 'Click to upload MRI scan'}
                  </p>
                  <p className="text-white/60 text-sm">
                    Supports JPG, PNG, GIF up to 10MB
                  </p>
                </label>
              </div>

              {/* Image Preview */}
              {preview && (
                <div className="mt-6">
                  <h4 className="text-white text-lg font-semibold mb-4">Preview:</h4>
                  <div className="relative">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full max-h-48 object-contain rounded-lg border border-white/20"
                    />
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setPreview(null);
                      }}
                      className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Symptoms Form */}
            <div>
              <h3 className="text-white text-xl font-semibold mb-6 flex items-center">
                <Stethoscope className="w-6 h-6 mr-2" />
                Clinical Symptoms
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    <Droplets className="w-4 h-4 inline mr-2" />
                    Oxygen Level (%)
                  </label>
                  <input
                    type="number"
                    value={symptoms.Oxygen_Level}
                    onChange={(e) => handleSymptomChange('Oxygen_Level', e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 95"
                    min="0"
                    max="100"
                  />
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    <Thermometer className="w-4 h-4 inline mr-2" />
                    Body Temperature (°F)
                  </label>
                  <input
                    type="number"
                    value={symptoms.Body_Temperature}
                    onChange={(e) => handleSymptomChange('Body_Temperature', e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 98.6"
                    min="90"
                    max="110"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    <Activity className="w-4 h-4 inline mr-2" />
                    Glucose Level (mg/dL)
                  </label>
                  <input
                    type="number"
                    value={symptoms.Glucose_Level}
                    onChange={(e) => handleSymptomChange('Glucose_Level', e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 100"
                    min="0"
                    max="500"
                  />
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    <Heart className="w-4 h-4 inline mr-2" />
                    Blood Pressure (mmHg)
                  </label>
                  <input
                    type="number"
                    value={symptoms.Blood_Pressure}
                    onChange={(e) => handleSymptomChange('Blood_Pressure', e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 120"
                    min="60"
                    max="200"
                  />
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    <FeverIcon className="w-4 h-4 inline mr-2" />
                    Fever Present
                  </label>
                  <div className="flex gap-4">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="fever"
                        value="1"
                        checked={symptoms.Fever === 1}
                        onChange={(e) => handleSymptomChange('Fever', parseInt(e.target.value))}
                        className="mr-2"
                      />
                      <span className="text-white">Yes</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="fever"
                        value="0"
                        checked={symptoms.Fever === 0}
                        onChange={(e) => handleSymptomChange('Fever', parseInt(e.target.value))}
                        className="mr-2"
                      />
                      <span className="text-white">No</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 mt-8">
            <button
              onClick={handleSubmit}
              disabled={loading || (!selectedFile && !validateSymptoms())}
              className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-500 disabled:to-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-all flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5 mr-2" />
                  Analyze Complete Data
                </>
              )}
            </button>
            <button
              onClick={resetForm}
              className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-lg transition-colors"
            >
              Reset
            </button>
          </div>

          {/* Results */}
          {prediction && (
            <div className="mt-8">
              <h3 className="text-white text-lg font-semibold mb-4">Analysis Result:</h3>
              <div className={`p-6 rounded-lg ${
                prediction.prediction === 'Normal' 
                  ? 'bg-green-500/20 border border-green-500/50' 
                  : 'bg-red-500/20 border border-red-500/50'
              }`}>
                <div className="flex items-start">
                  {prediction.prediction === 'Normal' ? (
                    <CheckCircle className="w-8 h-8 text-green-400 mr-3 mt-1" />
                  ) : (
                    <AlertTriangle className="w-8 h-8 text-red-400 mr-3 mt-1" />
                  )}
                  <div className="flex-1">
                    <p className={`text-xl font-bold ${
                      prediction.prediction === 'Normal' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {prediction.prediction}
                    </p>
                    <p className={`text-sm mt-1 ${
                      prediction.prediction === 'Normal' ? 'text-green-300' : 'text-red-300'
                    }`}>
                      Confidence: <span className={getConfidenceColor(prediction.confidence)}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </p>
                    <p className="text-white/80 text-sm mt-2">
                      {prediction.prediction === 'Normal' 
                        ? 'No signs of encephalitis detected. Continue monitoring symptoms.' 
                        : 'Potential signs of encephalitis detected. Please consult a healthcare professional immediately.'
                      }
                    </p>
                    <div className="mt-3 text-xs text-white/60">
                      <p>Analysis based on: {prediction.from.image ? 'MRI Scan' : ''} {prediction.from.image && prediction.from.symptoms ? ' + ' : ''} {prediction.from.symptoms ? 'Clinical Symptoms' : ''}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/20 border border-red-500/50 text-red-300 px-4 py-3 rounded-lg mt-6">
              {error}
            </div>
          )}

          {/* Disclaimer */}
          <div className="text-center text-white/60 text-sm mt-6">
            <p>
              ⚠️ This tool is for educational and research purposes only. 
              Always consult with qualified healthcare professionals for medical diagnosis and treatment.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 