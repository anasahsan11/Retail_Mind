
import joblib
import pandas as pd

from .config import MODEL_DIR
from .features import engineer_features


# Model paths for each development stage
MODEL_PATHS = {
    "day1": MODEL_DIR / "churn_pipeline.joblib",
    "day2": MODEL_DIR / "churn_champion_day2.joblib",
    "day3": MODEL_DIR / "churn_v3.joblib",
    "day4": MODEL_DIR / "churn_v4.joblib",
}


def predict(customer, model_version="day3"):

    # 1. Validate model version
    if model_version not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model version: {model_version}. "
            f"Available versions: {list(MODEL_PATHS.keys())}"
        )

    # 2. Check model availability
    model_path = MODEL_PATHS[model_version]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the model first."
        )

    # 3. Load trained model
    model = joblib.load(model_path)

    # 4. Handle single or batch prediction
    is_batch = isinstance(customer, list)
    records = customer if is_batch else [customer]

    if not records:
        return []

    # 5. Convert input into DataFrame
    frame = pd.DataFrame(records)

    # 6. Convert TotalCharges to numeric
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"],
            errors="coerce"
        )

    # 7. Apply Day 4 feature engineering
    if model_version == "day4":
        frame = engineer_features(frame)

    # 8. Generate predictions
    predictions = model.predict(frame)

    # 9. Calculate churn probabilities
    classes = list(model.classes_)

    positive_label = 1 if 1 in classes else "Yes"
    churn_index = classes.index(positive_label)

    probabilities = model.predict_proba(frame)[:, churn_index]

    # 10. Format results
    results = []

    for label, probability in zip(predictions, probabilities):

        if model_version == "day4":
            prediction_label = "Yes" if label == 1 else "No"
        else:
            prediction_label = str(label)

        results.append({
            "prediction": prediction_label,
            "churn_probability": float(probability),
            "model_version": model_version,
        })

    return results if is_batch else results[0]
