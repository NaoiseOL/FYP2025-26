import React, { useState } from "react";
import "./images.css";
import PredictButton from "../predictButton";
import { fetchPreds, createPred } from "../../api";

function ImageUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);

    function handleChange(e) {
        const selected = e.target.files[0];
        setFile(selected);
        setPreview(URL.createObjectURL(selected));
    }

    async function handlePredict() {
        try {
            const prediction = await createPred(file);
            setResult(prediction);
        } catch (err) {
            console.error("Prediction failed:", err);
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
                        label="Predict Image"
                        onClick={handlePredict}
                    />
                </>
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