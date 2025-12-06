import React from 'react';
import './privacy.css'; // Custom CSS

const Privacy = () => {

  const acceptPolicy = () => {
    alert("Privacy policy accepted.");
    window.location.href = "/"; // Redirect to home after acceptance
  };

  return (
    <div className="privacy-page">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a href="/upload">Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/privacy" className="active">Privacy</a></li>
        </ul>
      </header>

      <div className="privacy-container">
        <h2>Privacy Policy</h2>
        <p>We value your privacy. This Privacy Policy explains how we collect, use, and protect your personal information.</p>

        <h3>Information We Collect</h3>
        <p>We may collect information such as uploaded MRI images, usage data, and browser metadata for diagnostic and improvement purposes.</p>

        <h3>How We Use Your Information</h3>
        <p>Your data is used strictly for analysis, prediction, and enhancing model accuracy. We do not sell or share your data with third parties.</p>

        <h3>Data Storage & Security</h3>
        <p>Uploaded images are processed securely and deleted automatically after the session.</p>

        <h3>Your Consent</h3>
        <p>By using our application, you consent to the terms outlined in this Privacy Policy.</p>

        <h3>Changes to This Policy</h3>
        <p>We may update this policy periodically. Changes will be reflected on this page.</p>

        <button className="accept-btn" onClick={acceptPolicy}>Accept & Continue</button>
      </div>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default Privacy;
