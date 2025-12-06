import React, { useState, useEffect } from 'react';
import './Results.css';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { selectedImage, tumorType, tumorConf, segmentedImage } = location.state || {};

  const [imageData, setImageData] = useState(null);
  const [classificationDetails, setClassificationDetails] = useState(null);
  const [segmentedImageUrl, setSegmentedImage] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [otherDetails, setOtherDetails] = useState('');

  useEffect(() => {
    // Redirect to upload page if no data
    if (!selectedImage && !tumorType) {
      navigate('/upload');
      return;
    }

    if (selectedImage) setImageData(selectedImage);

    if (tumorType && tumorConf !== undefined) {
      setClassificationDetails({
        tumorType,
        confidence: `${tumorConf}%`
      });
    }

    if (segmentedImage) {
      // Construct full URL if backend returns relative path
      setSegmentedImage(segmentedImage);
    }
  }, [selectedImage, tumorType, tumorConf, segmentedImage, navigate]);

  const goToUploadPage = () => {
    navigate('/upload');
  };

  const saveResults = async () => {
    if (!patientName || !otherDetails) {
      alert('Please enter the patient name and other details.');
      return;
    }

    const formData = new FormData();
    formData.append('patientName', patientName);
    formData.append('tumorType', classificationDetails.tumorType);
    formData.append('confidence', classificationDetails.confidence);
    formData.append('otherDetails', otherDetails);

    try {
      // Append original image
      if (imageData) {
        const originalImageBlob = await fetch(imageData).then(res => res.blob());
        const originalImageFile = new File([originalImageBlob], 'originalImage.png', { type: 'image/png' });
        formData.append('originalImage', originalImageFile);
      }

      // Append segmented image
      if (segmentedImageUrl) {
        const segmentedImageBlob = await fetch(segmentedImageUrl).then(res => res.blob());
        const segmentedImageFile = new File([segmentedImageBlob], 'segmentedImage.png', { type: 'image/png' });
        formData.append('segmentedImage', segmentedImageFile);
      }

      const response = await axios.post('http://localhost:5000/save-results', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.status === 200) {
        alert('Results saved successfully!');
        setShowModal(false);
      } else {
        alert('Error saving results.');
      }
    } catch (error) {
      console.error('Error saving results:', error);
      alert('There was an error saving the results.');
    }
  };

  return (
    <div className="results">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a href="/upload">Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a href="/about">About</a></li>
        </ul>
      </header>

      <section className="results-container">
        <h2>Analysis Results</h2>

        <div className="results-content">
          <div className="image-section">
            <h3>Original MRI Image</h3>
            {imageData ? (
              <div className="image-container">
                <img src={imageData} alt="Uploaded MRI" className="preview-image" />
              </div>
            ) : (
              <p>No image uploaded.</p>
            )}
          </div>

          <div className="segmented-image-section">
            <h3>Segmented Image</h3>
            {segmentedImageUrl ? (
              <div className="image-container">
                <img src={segmentedImageUrl} alt="Segmented MRI" className="preview-image" />
              </div>
            ) : (
              <p>No segmented image available.</p>
            )}
          </div>
        </div>

        <div className="classification-details">
          {classificationDetails && (
            <div className="details-container">
              <h4>Tumor Type: {classificationDetails.tumorType}</h4>
              <p>Confidence: {classificationDetails.confidence}</p>
            </div>
          )}
        </div>

        <div className="buttons">
          <button onClick={goToUploadPage} className="back-btn">Upload Next</button>
          <button onClick={() => setShowModal(true)} className="save-btn">Save Results</button>
        </div>
      </section>

      {showModal && (
        <div className="modal">
          <div className="modal-content">
            <h3>Enter Patient Details</h3>
            <form onSubmit={(e) => { e.preventDefault(); saveResults(); }}>
              <div>
                <label htmlFor="patientName">Patient Name</label>
                <input
                  type="text"
                  id="patientName"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="Enter patient name"
                  required
                />
              </div>
              <div>
                <label htmlFor="otherDetails">Other Details</label>
                <textarea
                  id="otherDetails"
                  value={otherDetails}
                  onChange={(e) => setOtherDetails(e.target.value)}
                  placeholder="Enter additional details"
                  required
                />
              </div>
              <div className="modal-buttons">
                <button type="button" onClick={() => setShowModal(false)} className="cancel-btn">Cancel</button>
                <button type="submit" className="save-btn">Save</button>
              </div>
            </form>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default Results;
