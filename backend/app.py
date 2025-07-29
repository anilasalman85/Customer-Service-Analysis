from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Union
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', './model/csat_predictor_model.joblib')
LABEL_ENCODERS_PATH = os.getenv('LABEL_ENCODERS_PATH', './model/label_encoders.joblib')

# Load the trained model and label encoders
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    logger.info("Successfully loaded model and label encoders")
except Exception as e:
    logger.error(f"Error loading model or label encoders: {str(e)}")
    raise

# The features used in the model
FEATURES = [
    'channel_name', 'category', 'Sub-category', 'Item_price', 
    'connected_handling_time', 'response_delay', 'survey_delay', 
    'sentiment_score', 'Agent_name', 'Agent Shift', 'Tenure Bucket'
]

def validate_input(data: Dict[str, Any]) -> List[str]:
    """
    Validate input data for required features and data types.
    
    Args:
        data: Dictionary containing input features
        
    Returns:
        List of validation error messages, empty if valid
    """
    errors = []
    
    # Check for required features
    for feature in FEATURES:
        if feature not in data:
            errors.append(f"Missing required feature: {feature}")
    
    # Validate numeric features
    numeric_features = ['Item_price', 'connected_handling_time', 'response_delay', 
                       'survey_delay', 'sentiment_score']
    for feature in numeric_features:
        if feature in data and not isinstance(data[feature], (int, float)):
            errors.append(f"Feature {feature} must be numeric")
    
    return errors

def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess input data for model prediction.
    
    Args:
        data: Dictionary of features
        
    Returns:
        np.array of processed features ready for model prediction
        
    Raises:
        ValueError: If preprocessing fails
    """
    try:
        df = pd.DataFrame([data])
        
        # Encode categorical columns using loaded label encoders
        for col in label_encoders:
            if col in df.columns:
                le = label_encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                # If 'Unknown' not in classes, add it dynamically
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                df[col] = le.transform(df[col])
        
        # Ensure all features are present, fill missing numerics with 0
        for feature in FEATURES:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns
        df = df[FEATURES]
        return df.values
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise ValueError(f"Preprocessing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")  # Rate limit for this specific endpoint
def predict():
    """
    Predict CSAT score based on input features.
    
    Returns:
        JSON response with prediction, probability, and label
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        input_data = request.json
        
        # Validate input
        validation_errors = validate_input(input_data)
        if validation_errors:
            return jsonify({"error": "Validation failed", "details": validation_errors}), 400
        
        # Process and predict
        processed = preprocess_input(input_data)
        pred = model.predict(processed)[0]
        pred_prob = model.predict_proba(processed)[0,1]
        
        # Log prediction
        logger.info(f"Prediction made: {pred} with probability {pred_prob:.2f}")
        
        return jsonify({
            "prediction": int(pred),
            "probability": float(pred_prob),
            "label": "High CSAT" if pred == 1 else "Low CSAT",
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
