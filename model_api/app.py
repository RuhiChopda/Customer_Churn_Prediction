from flask import Flask, request, jsonify
import joblib, pathlib, pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load(pathlib.Path(__file__).resolve().parents[1] / 'model_api' / 'churn_logistic.pkl')
scaler = joblib.load(pathlib.Path(__file__).resolve().parents[1] / 'model_api' / 'scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df['gender'] = (df['gender']=='Male').astype(int)
    df['Partner'] = (df['Partner']=='Yes').astype(int)
    df['Dependents'] = (df['Dependents']=='Yes').astype(int)
    df['PhoneService'] = (df['PhoneService']=='Yes').astype(int)
    features = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MonthlyCharges','TotalCharges']
    X = df[features].fillna(0)
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[0,1])
    return jsonify({'churn_probability': prob})

@app.route('/health', methods=['GET'])
def health():
    logger.info('Health check OK')
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
