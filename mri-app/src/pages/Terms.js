import React from 'react';
import './Terms.css';

const Terms = () => {
  return (
    <div className="terms-page">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a href="/upload">Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/terms" className="active">Terms</a></li>
        </ul>
      </header>

      <div className="terms-container">
        <h2>Terms & Conditions</h2>
        <p>Welcome to our MRI AI application. By using this service, you agree to comply with and be bound by the following terms and conditions:</p>

        <h3>1. Use of the Application</h3>
        <p>The platform is designed for educational and diagnostic purposes only. It should not replace professional medical advice.</p>

        <h3>2. Data Usage</h3>
        <p>Any MRI images uploaded are used strictly for analysis and model improvement. Your data will remain confidential and will not be sold to third parties.</p>

        <h3>3. User Responsibilities</h3>
        <p>Users must provide accurate information and ensure they have consent to upload any MRI data. Misuse of the platform may result in termination of access.</p>

        <h3>4. Intellectual Property</h3>
        <p>All content, code, and AI models are property of the application owners. Unauthorized copying or distribution is prohibited.</p>

        <h3>5. Limitation of Liability</h3>
        <p>The platform provides predictive analysis and segmentation results. The owners are not liable for medical decisions made based on these results.</p>

        <h3>6. Changes to Terms</h3>
        <p>We may update these terms periodically. Continued use of the application implies acceptance of any changes.</p>
      </div>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default Terms;
