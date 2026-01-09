import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

# Get the project root directory (one level up from this script)
BASE_DIR = Path(__file__).resolve().parent.parent

def train_and_save():
    # 1. Load data
    data_path = BASE_DIR / "data" / "sensor_data.csv"

    print("Starting data loading...")
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from: {data_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file at {data_path}")
        print("Error: sensor_data.csv not found, please run the data generation script first!")
        return

    # 2. Define features and target
    # Ensure these column names match your data generation script exactly
    features = ['Vibration', 'Temperature', 'Pressure', 'OperatingHours', 'Vibration_Mean']
    target_cls = 'Failure'
    target_reg = 'RUL'

    X = df[features]
    y_cls = df[target_cls]
    y_reg = df[target_reg]

    # 3. Split training and testing sets (for model validation)
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # 4. Train Random Forest classifier (RF) / Random Forest Regressor
    print("Training Random Forest model...")
    rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_clf.fit(X_train, y_train_cls)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)

    # Output brief evaluation results
    y_pred_cls = rf_clf.predict(X_test)
    print("\nClassification Model Evaluation Report:")
    print(classification_report(y_test_cls, y_pred_cls))

    y_pred_reg = rf_reg.predict(X_test_reg)
    print("\nRegression Model Evaluation Report:")
    print(f"(MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.2f} 小时")
    print(f"(R2 Score): {r2_score(y_test_reg, y_pred_reg):.2f}")

    # 5. Train Anomaly Detection model (Isolation Forest)
    print("Training Anomaly Detection model...")
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)  # Anomaly detection usually uses the full dataset for feature baseline learning

    # 6. Save models to local files
    print("\nSaving models...")

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(rf_clf, model_dir / 'rf_model.pkl')
    joblib.dump(rf_reg, model_dir / 'rf_reg_model.pkl')
    joblib.dump(iso_model, model_dir / 'iso_model.pkl')
    # Recommendation: save the feature list to prevent mismatching order during inference
    joblib.dump(features, model_dir / 'feature_names.pkl')

    print(f"Success! Models saved in: {model_dir}")


if __name__ == "__main__":
    train_and_save()