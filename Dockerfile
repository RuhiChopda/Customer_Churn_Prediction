FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pandas scikit-learn flask joblib
EXPOSE 5000
CMD ["gunicorn", "model_api.app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
