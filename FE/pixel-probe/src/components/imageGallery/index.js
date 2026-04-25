import React, { useEffect, useState } from "react";
import { fetchPreds } from "../../api";
import { useNavigate } from "react-router-dom";
import "./imageGallery.css";

function MyGallery() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const S3_BASE_URL = "https://pixel-probe-images.s3.us-east-1.amazonaws.com/";

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please log in to view your predictions.");
      setLoading(false);
      return;
    }

    async function loadData() {
      try {
        const result = await fetchPreds();
        setData(result);
      } catch (err) {
        setError("Failed to load predictions. Please try again.");
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div className="gallery-page">
      <h1>Predictions</h1>

      {loading ? (
        <h4 className="loading-text">Loading Data...</h4>
      ) : error ? (
        <div>
          <h4 className="loading-text" style={{ color: "#e74c3c" }}>
            {error}
          </h4>
          {!localStorage.getItem("token") && (
            <button
              className="logout-button"
              onClick={() => navigate("/")}
            >
              Go to Login
            </button>
          )}
        </div>
      ) : data.length === 0 ? (
        <h4 className="loading-text">No predictions yet.</h4>
      ) : (
        <div className="gallery-container">
          {data.map((item) => {
            const imageUrl = `${S3_BASE_URL}${item.image_name}`;
            return (
              <div key={item.pred_id} className="gallery-card">
                <img src={imageUrl} alt={item.image_name} />
                <div className="gallery-info">
                  <div>
                    <b>Name:</b> {item.image_name}
                  </div>
                  <div>
                    <b>Prediction:</b> {item.prediction}
                  </div>
                  <div>
                    <b>Confidence:</b> {(item.confidence * 100).toFixed(2)}%
                  </div>
                  <div>
                    <b>Date:</b> {item.date_time}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default MyGallery;
