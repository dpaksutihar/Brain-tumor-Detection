import React, { useState, useEffect } from 'react';
import './Results.css';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

// --- Static Color and Label Mapping for Segmentation Scores ---
const SEG_DETAILS_MAP = [
    { level: 'High Confidence Area', color: '#FF0000', className: 'high-conf', scoreIndex: 0 }, // Index 0: Red
    { level: 'Medium Confidence Area', color: '#FF00FF', className: 'medium-conf', scoreIndex: 1 }, // Index 1: Magenta
    { level: 'Low Confidence Area', color: '#00FFFF', className: 'low-conf', scoreIndex: 2 } // Index 2: Cyan
];

// --- New Color Scheme for Overall Segmentation Bar (Bar 1) ---
const SEG_COLOR_RANGES = [
    { min: 0, max: 50, color: 'green', label: 'Low Match' },
    { min: 50, max: 70, color: '#00FFFF', label: 'Moderate Match' }, // Cyan
    { min: 70, max: 90, color: '#794a9d', label: 'Good Match' }, // Magenta
    { min: 90, max: 100, color: '#FF0000', label: 'Excellent Match' }, // Red
];

const Results = () => {
    const navigate = useNavigate();
    const location = useLocation();
    // Expecting segConf to be an array of three scores: [high, medium, low]
    const { selectedImage, tumorType, tumorConf, segmentedImage, segConf } = location.state || {};

    const [imageData, setImageData] = useState(null);
    const [analysisDetails, setAnalysisDetails] = useState(null);
    const [segmentedImageUrl, setSegmentedImage] = useState(null);
    const [showSaveModal, setShowSaveModal] = useState(false);
    const [showZoomModal, setShowZoomModal] = useState(false);
    const [zoomedImageSrc, setZoomedImageSrc] = useState(null);
    const [patientName, setPatientName] = useState('');
    const [otherDetails, setOtherDetails] = useState('');

    useEffect(() => {
        if (!selectedImage && !tumorType) {
            navigate('/upload');
            return;
        }

        if (selectedImage) setImageData(selectedImage);

        if (tumorType && tumorConf !== undefined) {
            setAnalysisDetails({
                tumorType,
                tumorConf: `${tumorConf}%`,
                // Ensure segConf is an array, map to floats, default to empty
                segConfList: Array.isArray(segConf) ? segConf.map(s => parseFloat(s)) : [], 
            });
        }

        if (segmentedImage) {
            setSegmentedImage(segmentedImage);
        }
    }, [selectedImage, tumorType, tumorConf, segmentedImage, segConf, navigate]);

    const goToUploadPage = () => {
        navigate('/upload');
    };

    const saveResults = async () => {
        if (!patientName || !otherDetails || !analysisDetails) {
            alert('Please complete all required data fields.');
            return;
        }

        const formData = new FormData();
        formData.append('patientName', patientName);
        formData.append('tumorType', analysisDetails.tumorType);
        formData.append('confidence', analysisDetails.tumorConf);
        formData.append('otherDetails', otherDetails);

        // Append segmentation details as a JSON string
        formData.append('segmentationScores', JSON.stringify(analysisDetails.segConfList)); 

        try {
            // Logic for fetching and appending images remains the same
            if (imageData) {
                const originalImageBlob = await fetch(imageData).then(res => res.blob());
                const originalImageFile = new File([originalImageBlob], 'originalImage.png', { type: 'image/png' });
                formData.append('originalImage', originalImageFile);
            }

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
                setShowSaveModal(false);
            } else {
                alert('Error saving results.');
            }
        } catch (error) {
            console.error('Error saving results:', error);
            alert('There was an error saving the results.');
        }
    };

    const zoomImage = (src) => {
        setZoomedImageSrc(src);
        setShowZoomModal(true);
    };

    const closeZoomModal = () => {
        setShowZoomModal(false);
        setZoomedImageSrc(null);
    };
    
    // Helper to get Classification Bar Color
    const getClassificationColor = (conf) => {
        if (analysisDetails.tumorType === 'No Tumour') return '#004d40'; // Dark Green for No Tumor
        if (conf >= 80) return '#c62828'; // Red for High Tumor Confidence
        if (conf >= 50) return '#FFB300'; // Amber for Medium Tumor Confidence
        return '#00FFFF'; // Cyan for Low Tumor Confidence
    };
    
    // Helper to calculate Overall Segmentation Score and Color (Bar 1 logic)
    const getOverallSegmentationScore = (segConfList) => {
        if (!segConfList || segConfList.length !== 3) return { score: 0, color: '#95a5a6', label: 'N/A' };

        // Calculate the overall Segmentation Score (Average of the three scores)
        const totalScore = segConfList.reduce((acc, score) => acc + score, 0);
        const averageScore = Math.min(100, totalScore / segConfList.length); // Cap at 100%

        let segmentColor = '#95a5a6'; // Default gray
        let segmentLabel = 'N/A';
        
        for (const range of SEG_COLOR_RANGES) {
            if (averageScore > range.min && averageScore <= range.max) {
                segmentColor = range.color;
                segmentLabel = range.label;
                break;
            }
        }
        
        return {
            score: averageScore.toFixed(1),
            color: segmentColor,
            label: `Overall Dice/IoU: ${averageScore.toFixed(1)}% (${segmentLabel})`,
        };
    };
    
    // Extract calculated data
    const tumorConfValue = analysisDetails?.tumorConf ? parseFloat(analysisDetails.tumorConf.replace('%', '')) : 0;
    const overallSegData = analysisDetails?.segConfList ? getOverallSegmentationScore(analysisDetails.segConfList) : { score: 0, color: '#95a5a6', label: 'N/A' };
    
    // --- JSX Rendering ---
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
                <h2>🧠 Brain Tumor Analysis Report</h2>

                {/* --- New Top-Level Layout Grid: 40% (Images) | 60% (Analysis) --- */}
                <div className="report-main-grid">
                    
                    {/* ===== LEFT COLUMN (40%): Images ===== */}
                    <div className="report-column-left">
                        
                        {/* Original Image Section (Top Left) */}
                        <div className="image-card"> 
                            <h3>Uploaded MRI Image</h3>
                            {imageData ? (
                                <div 
                                    className="image-preview-wrapper"
                                    onClick={() => zoomImage(imageData)}
                                    title="Click to Zoom"
                                >
                                    <img src={imageData} alt="Uploaded MRI" className="preview-image" />
                                </div>
                            ) : (
                                <p className="image-placeholder-text">No image uploaded.</p>
                            )}
                        </div>

                        {/* Segmented Image Section (Bottom Left) */}
                        <div className="image-card"> 
                            <h3>Segmentation Overlay </h3>
                            {segmentedImageUrl ? (
                                <div 
                                    className="image-preview-wrapper image-segmented-highlight"
                                    onClick={() => zoomImage(segmentedImageUrl)}
                                    title="Click to Zoom"
                                >
                                    <img src={segmentedImageUrl} alt="Segmented MRI" className="preview-image" />
                                </div>
                            ) : (
                                <p className="image-placeholder-text">Segmentation N/A or No Tumor Detected.</p>
                            )}
                        </div>
                    </div>

                    {/* ===== RIGHT COLUMN (60%): Analysis and Buttons ===== */}
                    <div className="report-column-right">
                        
                        {/* --- ANALYSIS SUMMARY --- */}
                        <div className="classification-details-card professional-layout">
                            <h3>Analysis Summary</h3>
                            {analysisDetails ? (
                                <div className="summary-blocks-stacked">
                                    
                                    {/* --- BLOCK 1: Tumor Classification and Confidence (Top Right) --- */}
                                    <div className="summary-block block-classification">
                                        <h4>Tumor Classification Result</h4>
                                        
                                        <div className="summary-item item-full-width">
                                            <p className="label">Tumor Type Detected</p>
                                            <p className="value tumor-type-result">
                                                <span className={analysisDetails.tumorType === 'No Tumour' ? 'status-low' : 'status-high'}>
                                                    {analysisDetails.tumorType}
                                                </span>
                                            </p>
                                        </div>

                                        <div className="summary-item confidence-section item-full-width">
                                            <p className="label">Classification Confidence: {analysisDetails.tumorConf}</p>
                                            <div className="progress-bar-container">
                                                <div 
                                                    className="progress-fill" 
                                                    style={{ 
                                                        width: analysisDetails.tumorConf, 
                                                        backgroundColor: getClassificationColor(tumorConfValue)
                                                    }}
                                                ></div>
                                            </div>
                                            <span className="info-tag">Model Certainty</span>
                                        </div>
                                    </div>

                                    {/* --- BLOCK 2: Segmentation Scores (Middle Right) --- */}
                                    <div className="summary-block block-segmentation">
                                        <h4>Segmentation Accuracy and Breakdown</h4>
                                        
                                        {/* -------------------- BAR 1: OVERALL SCORE (Average, Dynamic Color) -------------------- */}
                                        <div className="summary-item confidence-section item-full-width">
                                            <p className="label">Overall Segmentation Score (Average): {overallSegData.score}%</p>
                                            <div className="progress-bar-container large-bar">
                                                <div 
                                                    className="progress-fill" 
                                                    style={{ 
                                                        width: `${overallSegData.score}%`, 
                                                        backgroundColor: overallSegData.color 
                                                    }}
                                                ></div>
                                            </div>
                                        </div>

                                        <hr className="small-separator" />
                                        
                                        {/* -------------------- BARS 2, 3, 4: INDIVIDUAL CONFIDENCE SCORES -------------------- */}
                                        <div className="individual-scores-container">
                                        {analysisDetails.segConfList.length === 3 ? (
                                            SEG_DETAILS_MAP.map((detail, index) => {
                                                const score = analysisDetails.segConfList[index];
                                                const scorePercentage = Math.min(100, Math.max(0, score)); // Cap score

                                                return (
                                                    <div className="individual-score-item" key={index}>
                                                        <p className="label score-label">
                                                            {detail.level.split('Area')[0].trim()}: 
                                                            <span className="score-value">
                                                                {scorePercentage.toFixed(1)}%
                                                            </span>
                                                        </p>
                                                        <div className="progress-bar-container small-bar">
                                                            <div 
                                                                className="progress-fill" 
                                                                style={{ 
                                                                    width: `${scorePercentage}%`, 
                                                                    backgroundColor: detail.color 
                                                                }}
                                                            ></div>
                                                        </div>
                                                    </div>
                                                );
                                            })
                                        ) : (
                                            <p>Individual segmentation confidence scores are not available.</p>
                                        )}
                                        </div>
                                    </div>
                                    
                                </div>
                            ) : (
                                <p>Awaiting analysis data...</p>
                            )}
                        </div>
                        
                        {/* Buttons (Bottom Right) */}
                        <div className="buttons">
                            <button onClick={goToUploadPage} className="back-btn">Upload Next ⬆️</button>
                            <button onClick={() => setShowSaveModal(true)} className="save-btn">Save Results 💾</button>
                        </div>
                        
                    </div>
                </div> {/* End of report-main-grid */}
                
            </section>
            
            {/* -------------------- Modals (Save & Zoom) remain the same -------------------- */}
            {showSaveModal && (
                <div className="modal-overlay">
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
                                <button type="button" onClick={() => setShowSaveModal(false)} className="cancel-btn">Cancel</button>
                                <button type="submit" className="save-btn">Save</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {showZoomModal && (
                <div className="zoom-modal-overlay" onClick={closeZoomModal}>
                    <div className="zoom-modal-content" onClick={e => e.stopPropagation()}>
                        <button className="zoom-close-btn" onClick={closeZoomModal}>&times;</button>
                        <img src={zoomedImageSrc} alt="Zoomed View" className="zoomed-image" />
                        <p className="zoom-caption">Click outside or press 'X' to close.</p>
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