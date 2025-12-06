import React from 'react';
import './About.css'; // For custom styles

const About = () => {
  return (
    <div className="about">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a href="/upload">Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a href="/about">About</a></li>
        </ul>
      </header>

      <div className="about-container">
        <h2>About MRI and AI in Medical Imaging</h2>
        <p>
          MRI (Magnetic Resonance Imaging) is a non-invasive medical imaging technique that uses strong magnetic fields and radio waves to generate detailed images of the internal structures of the body, particularly the brain.
        </p>
        <p>
          Our AI-driven platform enhances MRI analysis by automating the detection and segmentation of brain tumors. Using deep learning models, it can provide faster, more accurate results, assisting radiologists in making informed decisions.
        </p>
      </div>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default About;
