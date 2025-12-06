import React from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate for navigation
import './Home.css'; // For custom styles

const Home = () => {
  const navigate = useNavigate(); // Use useNavigate for navigation

  const navigateToUpload = () => {
    navigate('/upload'); // Redirect to Upload page
  };

  const navigateToAbout = () => {
    navigate('/about'); // Redirect to About page
  };

  return (
    <div className="home">
      <header className="navbar">
        <ul>
          <li><a href="/">Home</a></li>
          <li><a onClick={navigateToUpload}>Upload MRI</a></li>
          <li><a href="/results">Results</a></li>
          <li><a href="/patient-data">Patient Data</a></li>
          <li><a onClick={navigateToAbout}>About</a></li>
        </ul>
      </header>

      <section className="hero">
        <h1>AI-Driven Brain Tumor Detection</h1>
        <p>Enhancing radiologists' capabilities with cutting-edge AI technology.</p>
        <div className="buttons">
          <button onClick={navigateToUpload}>Get Started</button>
          <button onClick={navigateToAbout}>Learn More</button>
        </div>
      </section>

      <section className="features">
        <h2>Features</h2>
        <div className="feature-card" onClick={navigateToUpload}>AI-Powered Classification</div>
        <div className="feature-card" onClick={navigateToUpload}>Real-Time Segmentation</div>
        <div className="feature-card" onClick={navigateToUpload}>Secure Patient Data Management</div>
      </section>

      <footer className="footer">
        <p>© 2025 MRI App | <a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms</a></p>
      </footer>
    </div>
  );
};

export default Home;