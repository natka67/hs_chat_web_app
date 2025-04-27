import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import Intro from "./components/Intro";

function App() {
  const [description, setDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClassify = async () => {
    if (!description.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post("http://tarrifs-backend-app.azurewebsites.net/classify", { description });
      setResult(response.data);
    } catch (error) {
      console.error("Error classifying -", error);
      setResult({ error: "Failed to classify. Server error." });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setDescription("");
    setResult(null);
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <Intro />
        <div className="app-card">
          <h1 className="app-title">
            Product HS Code Classifier
          </h1>

          <div className="form-group">
            <label className="form-label">
              Product Description
            </label>
            <textarea
              rows="4"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter detailed product description..."
              className="form-textarea"
            />
          </div>

          <div className="button-container">
            <button
              onClick={handleClassify}
              disabled={loading}
              className="primary-button"
            >
              {loading ? "Processing..." : "Classify Product"}
            </button>

            <button
              onClick={handleClear}
              className="secondary-button"
            >
              Clear
            </button>

            <button
              onClick={() => window.open('http://tarrifs-backend-app.azurewebsites.net/download-pdfs', '_blank')}
              className="secondary-button"
            >
              Download Input PDF
            </button>
          </div>


          <div>
            <h2 className="result-title">
              Classification Result
            </h2>
            <div className={`result-box ${result ? 'with-result' : ''}`}>
              {result ? (
                result.error ? (
                  <p className="error-message">{result.error}</p>
                ) : (
                  <div>
                    <div className="result-row">
                      <span className="result-label">HS Code:</span>
                      <span className="result-value">{result.hs_code}</span>
                    </div>
                    <div className="result-row">
                      <span className="result-label">Confidence:</span>
                      <span>{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <div>
                      <span className="result-label">Notes:</span>
                      <p>{result.notes}</p>
                    </div>
                  </div>
                )
              ) : (
                <p className="empty-message">
                  No classification yet. Enter a product description and click "Classify Product".
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;