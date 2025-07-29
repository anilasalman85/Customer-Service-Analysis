import React, { useState } from "react";

const FEATURES = [
  "channel_name",
  "category",
  "Sub-category",
  "Item_price",
  "connected_handling_time",
  "response_delay",
  "survey_delay",
  "sentiment_score",
  "Agent_name",
  "Agent Shift",
  "Tenure Bucket",
];

// Numeric features for input type number
const numericFeatures = new Set([
  "Item_price",
  "connected_handling_time",
  "response_delay",
  "survey_delay",
  "sentiment_score",
]);

function App() {
  // Initialize form state with empty strings
  const [formData, setFormData] = useState(
    FEATURES.reduce((acc, f) => ({ ...acc, [f]: "" }), {})
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Update form state
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((fd) => ({ ...fd, [name]: value }));
  };

  // Submit handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    // Prepare payload - convert numeric fields to numbers (floats)
    const payload = {};
    for (const key of FEATURES) {
      if (numericFeatures.has(key)) {
        payload[key] = parseFloat(formData[key]);
      } else {
        payload[key] = formData[key];
      }
    }

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Prediction failed");
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", fontFamily: "Arial, sans-serif" }}>
      <h1>CSAT Score Predictor</h1>
      <form onSubmit={handleSubmit}>
        {FEATURES.map((feature) => (
          <div key={feature} style={{ marginBottom: 15 }}>
            <label htmlFor={feature} style={{ display: "block", fontWeight: "bold" }}>
              {feature}:
            </label>
            <input
              type={numericFeatures.has(feature) ? "number" : "text"}
              step={numericFeatures.has(feature) ? "any" : undefined}
              id={feature}
              name={feature}
              value={formData[feature]}
              onChange={handleChange}
              required
              style={{
                width: "100%",
                padding: 8,
                fontSize: 16,
                borderRadius: 4,
                border: "1px solid #ccc",
              }}
            />
          </div>
        ))}

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 20px",
            fontSize: 16,
            cursor: loading ? "not-allowed" : "pointer",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
          }}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 20, color: "red", fontWeight: "bold" }}>
          Error: {error}
        </div>
      )}

      {result && (
        <div
          style={{
            marginTop: 20,
            padding: 15,
            border: "1px solid #007bff",
            borderRadius: 4,
            backgroundColor: "#e7f1ff",
          }}
        >
          <h2>Prediction Result</h2>
          <p>
            <strong>Prediction:</strong> {result.label} ({result.prediction})
          </p>
          <p>
            <strong>Probability:</strong> {(result.probability * 100).toFixed(2)}%
          </p>
          <p>
            <strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
