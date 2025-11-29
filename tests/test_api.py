import json
from model_api import app as flask_app
def test_predict_endpoint():
    client = flask_app.test_client()
    payload = {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 5,
      "PhoneService": "Yes",
      "MonthlyCharges": 90,
      "TotalCharges": 450
    }
    resp = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'churn_probability' in data
    assert 0.0 <= data['churn_probability'] <= 1.0
