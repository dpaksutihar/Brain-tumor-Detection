import React, { useState } from 'react';
import './MRIUpload.css'; 
import { useNavigate } from 'react-router-dom'; 
import axios from 'axios'; 

const MRIUpload = () => {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null); 
  const [isProcessing, setIsProcessing] = useState(false); 
  const [progress, setProgress] = useState(0); 
  const [isAnalyzing, setIsAnalyzing] = useState(false); 

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validImageTypes = ['image/jpeg', 'image/png'];
      if (validImageTypes.includes(file.type)) {
        setSelectedImage(URL.createObjectURL(file)); 
        setImageFile(file); // Store the actual file object
      } else {
        alert("Please upload a valid JPG or PNG image.");
      }
    }
  };

  // Send image for classification and segmentation analysis
  const startAnalysis = async () => {
    if (!imageFile) {
      alert("Please upload an MRI scan before starting analysis.");
      return;
    }

    setIsProcessing(true);
    setIsAnalyzing(true);
    setProgress(0);

    // Simulate progress steps visually while the backend works
    let steps = ['Noise Reduction', 'Pre-processing', 'Segmentation', 'Classification'];
    let stepIndex = 0;

    const processStep = () => {
      if (stepIndex < steps.length) {
        const newProgress = (stepIndex + 1) * 25; 
        setProgress(newProgress);
        stepIndex++;
        // If analysis is still running, keep animating
        if (stepIndex < steps.length) {
            setTimeout(processStep, 800);
        }
      }
    };
    
    processStep(); // Start visual progress
    sendImageForAnalysis(); // Start actual backend request
  };

  // Function to send image for analysis to the backend
  const sendImageForAnalysis = async () => {
    const formData = new FormData();

    // Directly append the file object from state. 
    // We name it 'file' here; the Python server will use this key.
    formData.append('file', imageFile); 

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status === 200) {
        const { tumorType, tumorConf,segConf, segmentedImage } = response.data;
        console.log("Full backend response:", response.data);

        setIsProcessing(false);
        // Navigate to results
        navigate('/results', {
          replace: true,
          state: {
            selectedImage,
            tumorType,
            tumorConf,
            segConf,
            segmentedImage,
          },
        });
      } else {
        throw new Error('Analysis failed');
      }

    } catch (error) {
      console.error('Error during analysis:', error);
      alert('There was an error processing the image. Please try again.');
      setIsProcessing(false);
      setIsAnalyzing(false);
      setProgress(0);
    }
  };

  const goHome = () => {
    navigate('/');
  };

  return (
    <div className="mri-upload">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a href="/upload">Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a href="/about">About</a></li>
        </ul>
      </header>

      <section className="upload-section">
        <h1>Upload MRI Scan 🧠</h1>
        <p>
          Start the brain tumor analysis by uploading a clear <strong>JPG</strong> or <strong>PNG</strong> MRI image. 
          For the best results, ensure the image is a proper <strong>medical scan (axial) with the prefrontal region positioned at the top</strong>.
        </p>

        <div className="upload-card-container">
            
            {/* Left Side: Upload Input */}
            <div className="upload-input-area">
                <input 
                    type="file" 
                    accept="image/jpeg, image/png" 
                    onChange={handleFileChange} 
                    id="file-upload"
                    className="upload-input-hidden"
                    disabled={isProcessing}
                />
                <label htmlFor="file-upload" className="upload-dropzone">
                    <span className="upload-icon">📁</span>
                    <span className="upload-text">Click to Select MRI Image</span>
                    <span className="upload-subtext">Supported formats: .png, .jpg</span>
                </label>
                <p className="file-status">
                    {imageFile ? `Selected: ${imageFile.name}` : "No file selected"}
                </p>
            </div>

            {/* Right Side: Preview */}
            {selectedImage && (
                <div className="upload-preview-area">
                    <h3>Preview</h3>
                    <div className="preview-box">
                        <img src={selectedImage} alt="MRI preview" className="preview-img" />
                    </div>
                </div>
            )}
        </div>

        {/* Progress Bar */}
        {isProcessing && (
          <div className="progress-container">
            <div className="progress-bar-wrapper">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
            <div className="progress-text">
                {progress < 100 ? `Processing... ${Math.round(progress)}%` : "Finalizing..."}
            </div>
          </div>
        )}

        <div className="buttons">
          <button onClick={goHome} className="back-btn" disabled={isProcessing}>Back to Home</button>
          <button 
            className="start-btn" 
            onClick={startAnalysis} 
            disabled={isProcessing || isAnalyzing || !imageFile}
          >
            {isProcessing ? 'Analyzing...' : 'Start Analysis 🚀'}
          </button>
        </div>
      </section>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default MRIUpload;