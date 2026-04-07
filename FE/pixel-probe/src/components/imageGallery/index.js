import React, { useEffect, useState } from "react";
import { fetchPreds } from "../../api";
import "./imageGallery.css";

function MyGallery() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const container = {
    display: "flex",
    flexDirection: "row",
    flexWrap: "wrap",
    margin: "4% auto",
  };

  // Your S3 bucket base URL
  const S3_BASE_URL = "https://pixel-probe-images.s3.us-east-1.amazonaws.com/";

  useEffect(() => {
    async function loadData() {
      try {
        const result = await fetchPreds();
        setData(result);
      } catch (err) {
        console.error("Error fetching predictions: ", err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div style={{ textAlign: "center", margin: "auto" }}>
      <h1 style={{ color: "green" }}>Predictions</h1>

      {loading ? (
        <h4>Loading Data...</h4>
      ) : (
        <div style={container}>
          {data.map((item) => {
            const imageUrl = `${S3_BASE_URL}${item.image_name}`;

            return (
              <div
                key={item.pred_id}
                style={{
                  minWidth: "30rem",
                  margin: "1% auto",
                  padding: "1%",
                  boxShadow: "0 2px 5px grey",
                  display: "flex",
                  fontSize: "larger",
                }}
              >
                {/* Image from S3 */}
                <img
                  src={imageUrl}
                  alt={item.image_name}
                  style={{
                    width: "150px",
                    height: "150px",
                    objectFit: "cover",
                    marginRight: "1rem",
                    borderRadius: "8px",
                  }}
                />

                {/* Text info */}
                <span style={{ textAlign: "left", margin: "auto" }}>
                  <div>
                    <b>Name: </b> {item.image_name}
                  </div>
                  <div>
                    <b>Prediction: </b> {item.prediction}
                  </div>
                  <div>
                    <b>Confidence: </b> {item.confidence}
                  </div>
                  <div>
                    <b>Date: </b> {item.date_time}
                  </div>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default MyGallery;
