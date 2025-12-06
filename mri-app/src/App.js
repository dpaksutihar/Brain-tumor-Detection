import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Upload from './pages/MRIUpload';
import About from './pages/About';
import Results from './pages/Results';
import PatientData from './pages/PatientData';
import Privacy from './pages/privacy';
import Terms from './pages/Terms';
 

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/about" element={<About />} />
        <Route path="/results" element={<Results />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/patient-data" element={<PatientData />} />
      </Routes>
    </Router>
  );
};

export default App;
