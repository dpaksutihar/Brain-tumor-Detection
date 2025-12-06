import React, { useState } from 'react';
import './MRIUpload.css'; // Custom styles for the MRI upload page
import { useNavigate } from 'react-router-dom'; // Navigation hook
import axios from 'axios'; // For API calls to the backend

const MRIUpload = () => {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null); // To send the image to the backend
  const [isProcessing, setIsProcessing] = useState(false); // For showing the progress bar
  const [progress, setProgress] = useState(0); // Progress bar state
  const [isAnalyzing, setIsAnalyzing] = useState(false); // Track if analysis has started

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check if file is a valid image (JPG, PNG)
      const validImageTypes = ['image/jpeg', 'image/png'];
      if (validImageTypes.includes(file.type)) {
        setSelectedImage(URL.createObjectURL(file)); // Create a URL for image preview
        setImageFile(file); // Save file to send later to the backend
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
    setIsAnalyzing(true);  // Mark analysis as started
    setProgress(0);

    let steps = ['Noise Reduction', 'Pre-processing', 'Segmenting', 'Classifying'];
    let stepIndex = 0;

    const processStep = () => {
      if (stepIndex < steps.length) {
        const newProgress = (stepIndex + 1) * 25; // Each step adds 25%
        setProgress(newProgress);
        stepIndex++;

        setTimeout(processStep, 1000);
      }
    };
    sendImageForAnalysis();
    processStep();
  };

  // Function to send image for analysis to the backend (commented out)
  const sendImageForAnalysis = async () => {

    const formData = new FormData();
    
    const originalImageBlob = await fetch(selectedImage).then(res => res.blob());

    // Create File objects from blobs
    const originalImageFile = new File([originalImageBlob], 'originalImage.png', { type: 'image/png' });

    // Append image files to the FormData object
    formData.append('originalImage', originalImageFile);

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status === 200) {
        const { tumorType, tumorConf, segmentedImage} = response.data;

        console.log("Full backend response:", response.data);

        setIsProcessing(false);
        navigate('/results', {
          state: {
            selectedImage,
            tumorType,
            tumorConf,
            segmentedImage,
            // segmentedImage,
          },
        });
      } else {
        alert('Error analyzing the MRI scan.');
        setIsProcessing(false);
        setIsAnalyzing(false);
        setProgress(0);
      }

    } catch (error) {
      console.error('Error during analysis:', error);
      alert('There was an error processing the image.');

      // 🔥 Fix: Reset state so retry works
      setIsProcessing(false);
      setIsAnalyzing(false);
      setProgress(0);
    }
  };

  // Navigate back to the home page
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
        <h1>Upload Your MRI Scan</h1>
        <p>Upload an MRI scan image for brain tumor classification and segmentation analysis.</p>

        <div className="upload-box">
          <input 
            type="file" 
            accept="image/jpeg, image/png" 
            onChange={handleFileChange} 
            id="file"
            className="upload-input"
          />
          <label htmlFor="file" className="upload-label">Select MRI Image</label>

          {selectedImage && (
            <div className="image-preview">
              <h3>Uploaded Image</h3>
              <div className="image-container">
                <img src={selectedImage} alt="MRI preview" className="preview-image" />
              </div>
            </div>
          )}
        </div>

        <div className="buttons">
          <button onClick={goHome} className="back-btn">Back to Home</button>
          <button className="start-btn" onClick={startAnalysis} disabled={isProcessing || isAnalyzing}>
            {isProcessing ? 'Processing...' : 'Start Analysis'}
          </button>
        </div>

        {isProcessing && (
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            <div className="progress-text">{`Processing: ${Math.round(progress)}%`}</div>
          </div>
        )}
      </section>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default MRIUpload;
