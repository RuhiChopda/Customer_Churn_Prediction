from pathlib import Path
def test_model_files_exist():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / 'model_api' / 'churn_logistic.pkl').exists()
    assert (repo_root / 'model_api' / 'scaler.pkl').exists()
