import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib, pathlib

data_path = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'churn_sample.csv'
df = pd.read_csv(data_path)
df['gender'] = (df['gender']=='Male').astype(int)
df['Partner'] = (df['Partner']=='Yes').astype(int)
df['Dependents'] = (df['Dependents']=='Yes').astype(int)
df['PhoneService'] = (df['PhoneService']=='Yes').astype(int)
X = df[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MonthlyCharges','TotalCharges']].fillna(0)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)
joblib.dump(clf, pathlib.Path(__file__).resolve().parents[1] / 'model_api' / 'churn_logistic.pkl')
joblib.dump(scaler, pathlib.Path(__file__).resolve().parents[1] / 'model_api' / 'scaler.pkl')
print('Training finished and model saved')
