import React, { useState, useEffect } from "react";
import axios from "axios";
import "./PatientData.css";

const PatientData = () => {
  const [patientRecords, setPatientRecords] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const fetchRecords = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-patient-data");

      if (response.status === 200) {
        const fixed = response.data.records.flat().map(rec => ({
          patientName: rec["Patient Name"],
          tumorType: rec["Tumor Type"] || "Not Specified",
          confidence: rec["Confidence"],
          otherDetails: rec["Other Details"] || "",
          date: rec["Date"] || "",
          time: rec["Time"] || "",
          originalImage: rec["Original Image"] ? rec["Original Image"].replace(/\\/g, "/") : "",
          segmentedImage: rec["Segmented Image"] ? rec["Segmented Image"].replace(/\\/g, "/") : "",
        }));
        setPatientRecords(fixed);
      }
    } catch (error) {
      console.error("Error fetching patient data:", error);
      alert("Unable to fetch patient records.");
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  const handleViewImages = (record) => {
    setSelectedRecord(record);
    setShowModal(true);
  };

  return (
    
    <div
      className="patient-data"
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh"
      }}
    >
      {/* Navbar */}
      <header className="navbar" style={{ backgroundColor: "#794a9d" }}>
        <ul style={{ margin: 0, padding: 0, display: "flex", listStyle: "none" }}>
          <li><a href="/" style={{ color: "#fff", textDecoration: "none" }}>Home</a></li>
          <li><a href="/upload" style={{ color: "#fff", textDecoration: "none" }}>Upload MRI</a></li>
          <li><a href="/results" style={{ color: "#fff", textDecoration: "none" }}>Results</a></li>
          <li><a href="/patient-data" className="active" style={{ color: "#fff", textDecoration: "none" }}>Patient Data</a></li>
          <li><a href="/about" style={{ color: "#fff", textDecoration: "none" }}>About</a></li>
        </ul>
      </header>

      {/* Main Content */}
      <section
        className="patient-data-container"
        style={{ flex: 1, padding: "20px", backgroundColor: "#f9f9f9" }}
      >
        <h2 style={{ color: "#794a9d" }}>Patient Records</h2>

        <button
          className="refresh-btn"
          style={{ backgroundColor: "#794a9d", color: "#fff", padding: "8px 12px", border: "none", borderRadius: "5px", cursor: "pointer", marginBottom: "15px" }}
          onClick={fetchRecords}
        >
          🔄 Refresh
        </button>

        {patientRecords.length === 0 ? (
          <p>No saved patient data found.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ backgroundColor: "#794a9d", color: "#fff" }}>
                <th style={{ padding: "10px" }}>Patient Name</th>
                <th style={{ padding: "10px" }}>Tumor Type</th>
                <th style={{ padding: "10px" }}>Confidence</th>
                <th style={{ padding: "10px" }}>Other Details</th>
                <th style={{ padding: "10px" }}>Date</th>
                <th style={{ padding: "10px" }}>Time</th>
                <th style={{ padding: "10px" }}>Images</th>
              </tr>
            </thead>
            <tbody>
              {patientRecords.map((record, index) => (
                <tr key={index} style={{ backgroundColor: index % 2 === 0 ? "#fff" : "#f2f2f2" }}>
                  <td style={{ padding: "8px" }}>{record.patientName}</td>
                  <td style={{ padding: "8px" }}>{record.tumorType}</td>
                  <td style={{ padding: "8px" }}>{record.confidence}</td>
                  <td style={{
                      padding: "8px",
                      whiteSpace: "pre-wrap",  // preserves line breaks
                      wordBreak: "break-word", // breaks long words if needed
                      maxWidth: "300px"        // optional: prevent stretching the table
                    }}
                  >
                    {record.otherDetails}
                  </td>
                  <td style={{ padding: "8px" }}>{record.date}</td>
                  <td style={{ padding: "8px" }}>{record.time}</td>
                  <td style={{ padding: "8px" }}>
                    {(record.originalImage || record.segmentedImage) ? (
                      <button
                        style={{ backgroundColor: "#794a9d", color: "#fff", padding: "6px 10px", border: "none", borderRadius: "5px", cursor: "pointer" }}
                        onClick={() => handleViewImages(record)}
                      >
                        Show Images
                      </button>
                    ) : "No Images"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Modal */}
      {showModal && selectedRecord && (
        <div
          className="modal"
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 999
          }}
        >
          <div
            className="modal-content"
            style={{
              backgroundColor: "#fff",
              padding: "20px",
              borderRadius: "8px",
              maxWidth: "900px",
              width: "90%",
            }}
          >
            <h3>Images for {selectedRecord.patientName}</h3>
            <div
              style={{
                display: "flex",
                flexDirection: "row",
                justifyContent: "center",
                alignItems: "flex-start",
                gap: "20px",
                flexWrap: "wrap",
              }}
            >
              {selectedRecord.originalImage && (
                <div style={{ textAlign: "center" }}>
                  <h4>Original MRI</h4>
                  <img
                    src={`${selectedRecord.originalImage}`}
                    alt="Original MRI"
                    style={{ maxWidth: "400px", borderRadius: "8px" }}
                  />
                </div>
              )}
              {selectedRecord.segmentedImage && (
                <div style={{ textAlign: "center" }}>
                  <h4>Segmented MRI</h4>
                  <img
                    src={`${selectedRecord.segmentedImage}`}
                    alt="Segmented MRI"
                    style={{ maxWidth: "400px", borderRadius: "8px" }}
                  />
                </div>
              )}
            </div>
            <button
              style={{ backgroundColor: "#794a9d", color: "#fff", marginTop: "15px", padding: "8px 12px", border: "none", borderRadius: "5px", cursor: "pointer" }}
              onClick={() => setShowModal(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer
        style={{
          backgroundColor: "#794a9d",
          color: "#fff",
          textAlign: "center",
          marginTop: "auto"
        }}
      >
        <p>
          © 2025 MRI App | <a href="/privacy" style={{ color: "#fff" }}>Privacy Policy</a> | <a href="/terms" style={{ color: "#fff" }}>Terms</a>
        </p>
      </footer>
    </div>
  );
};

export default PatientData;
