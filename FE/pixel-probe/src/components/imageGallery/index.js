import React from "react";
import { useEffect, useState } from "react";
import { fetchPreds } from "../../api";
import "./imageGallery.css"


function MyGallery () {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const container = {
      display: "flex",
      flexDirection: "row",
      flexWrap: "wrap",
      margin: "4% auto",
    };

    useEffect(() => {
      async function loadData() {
        try {
          const result = await fetchPreds();
          setData(result);
        } catch (err){
          console.error("Error fetching predictions: ", err);
        } finally {
          setLoading(false);
        }
      }

      loadData();
    }, []);

    return (
        <div
            style={{ textAlign: "center", margin: "auto" }}
        >
            <h1 style={{ color: "green" }}>
                Predictions
            </h1>
            <h3>
                Display values from database without
                reloading...
            </h3>
            {loading ? (
                <h4>Loading Data...</h4>
            ) : (
                <div style={container}>
                    {data.map((item) => {
                        return (
                            <div
                                key={item.pred_id}
                                style={{
                                    minWidth: "30rem",
                                    margin: "1% auto",
                                    padding: "1%",
                                    boxShadow:
                                        "0 2px 5px grey",
                                    display: "flex",
                                    fontSize: "larger",
                                    margin: "1% auto",
                                }}
                            >
                                <span
                                    style={{
                                        textAlign: "left",
                                        margin: "auto",
                                    }}

                                    // style={style}
                                >
                                    <div>
                                        <b>Name: </b>
                                        {
                                            item.image_name
                                        }
                                    </div>
                                    <div>
                                        <b>Prediction </b>{" "}
                                        {item.prediction}
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
