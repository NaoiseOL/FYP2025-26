import React, { useState } from "react";
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
            console.log("Prediction response:", prediction);
            setResult(prediction);
        } catch (err) {
            console.error("Prediction failed:", err); 
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
                <>
                    <img 
                        src={preview} 
                        alt="Uploaded preview" 
                        className="preview" 
                    />

                    <PredictButton 
                        label={loading ? "Predicting..." : "Predict Image"}
                        onClick={handlePredict}
                        disabled={loading}
                    />
                </>
            )}

            {error && (
                <p style={{ color: 'red' }}>Error: {error}</p>
            )}

            {result && (
                <div className="result">
                    <p>Prediction: {result.prediction}</p>
                    <p>Confidence: {result.confidence}</p>
                </div>
            )}
        </div>
    );
}

export default ImageUpload;