import React, { useState } from "react";
import "./images.css";
import PredictButton from "../predictButton";

function ImageUpload() {
    const [file, setFile] = useState(null);

    function handleChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
    }

    return (
        <div className="image-upload">
            <h2>Upload Image</h2>
            <input type="file" onChange={handleChange} />

            {file && (
                <>
                    <img 
                        src={file} 
                        alt="Uploaded preview" 
                        className="preview" 
                    />

                    <PredictButton 
                        label="Predict Image"
                        onClick={() => console.log("Predicting image...")}
                    />
                </>
            )}
        </div>
    );
}

export default ImageUpload;
