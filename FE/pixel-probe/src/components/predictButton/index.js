import React from "react";
import "./predictButton.css"

function predictButton({ label, onClick }) {
  return (
    <button 
      className="predict-btn"
      onClick={onClick}
    >
      {label}
    </button>
  );
}

export default predictButton;
