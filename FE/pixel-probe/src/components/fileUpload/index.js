import React, { useState } from "react";
import GaugeComponent from "react-gauge-component";
import "./images.css";
import PredictButton from "../predictButton";
import { createPred } from "../../api";

function ImageUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleChange(e) {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  async function handlePredict() {
    try {
      setLoading(true);
      setError(null);
      const prediction = await createPred(file);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="image-upload">
      <h2>Upload Image</h2>

      <input type="file" onChange={handleChange} />

      {preview && (
        <div className={`content-row ${result ? "two-col" : ""}`}>
          <div className="preview-section">
            <img src={preview} alt="Uploaded preview" className="preview" />

            <div className="predict-btn-wrapper">
              <PredictButton
                label={loading ? "Predicting..." : "Predict Image"}
                onClick={handlePredict}
                disabled={loading}
              />
            </div>
          </div>

          {result && (
            <div className="result-card">
              <h3>Prediction Result</h3>
              <p className="prediction-text">{result.prediction}</p>

              <div className="gauge-wrapper">
                <GaugeComponent
                  value={result.confidence * 100}
                  arc={{
                    subArcs: [
                      { limit: 33, color: "#FF4C4C" },
                      { limit: 66, color: "#FFD93D" },
                      { limit: 100, color: "#4CAF50" },
                    ],
                    padding: 0.02,
                    width: 0.2,
                  }}
                  labels={{
                    valueLabel: { formatTextValue: (v) => `${v}%` },
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {error && <p className="error-text">Error: {error}</p>}
    </div>
  );
}

export default ImageUpload;
