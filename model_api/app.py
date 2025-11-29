from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging
from pathlib import Path

app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model & scaler from ROOT (your repo structure)
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'churn_logistic.pkl')
scaler = joblib.load(BASE_DIR / 'scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        df['gender'] = (df['gender'] == 'Male').astype(int)
        df['Partner'] = (df['Partner'] == 'Yes').astype(int)
        df['Dependents'] = (df['Dependents'] == 'Yes').astype(int)
        df['PhoneService'] = (df['PhoneService'] == 'Yes').astype(int)

        features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MonthlyCharges', 'TotalCharges'
        ]

        X = df[features].fillna(0)
        X_scaled = scaler.transform(X)

        prob = float(model.predict_proba(X_scaled)[0][1])
        return jsonify({'churn_probability': prob})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check OK")
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
