
import json
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from .config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from .data import load_dataset
from .features import engineer_features
from .pipeline import build_preprocessor


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    classes = list(model.classes_)
    positive_label = 1 if 1 in classes else "Yes"
    positive_index = classes.index(positive_label)

    probabilities = model.predict_proba(X)[:, positive_index]

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(
            y, predictions,
            pos_label=positive_label,
            zero_division=0
        )),
        "recall": float(recall_score(
            y, predictions,
            pos_label=positive_label,
            zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            (y == positive_label).astype(int),
            probabilities
        )),
    }


def make_day4_pipeline(X, classifier):

    preprocessor = build_preprocessor(X)

    feature_selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
        threshold="median",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("classifier", classifier),
    ])


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load original dataset
    X, y,_ = load_dataset()

    # 2. Preserve original features for Day 3 comparison
    X_original = X.copy()

    # 3. Engineer Day 4 features
    X_day4 = engineer_features(X)

    # 4. Encode target for Day 4 models
    y_day4 = (y == "Yes").astype(int)

    # 5. Create one consistent train/test split
    (
        X_train_raw,
        X_test_raw,
        y_train_raw,
        y_test_raw,
    ) = train_test_split(
        X_original,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_day4,
        y_day4,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_day4,
    )

    # 6. Cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "recall": make_scorer(recall_score, pos_label=1),
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    # 7. Define ensemble models and hyperparameter grids
    model_grids = {

        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [200, 300],
                "classifier__max_depth": [10, None],
                "classifier__min_samples_leaf": [2, 5],
                "classifier__max_features": ["sqrt", 0.8],
            },
        ),

        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
            },
        ),
    }

    results = {}
    fitted_models = {}

    # 8. Train and tune each model
    for model_name, (classifier, param_grid) in model_grids.items():

        print(f"\nTraining {model_name}...")

        pipeline = make_day4_pipeline(X_train, classifier)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="recall",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_

        train_recall = float(
            cv_results["mean_train_recall"][best_index]
        )

        validation_recall = float(
            cv_results["mean_test_recall"][best_index]
        )

        results[model_name] = {
            "best_params": search.best_params_,
            "cv_recall": float(search.best_score_),
            "cv_roc_auc": float(
                cv_results["mean_test_roc_auc"][best_index]
            ),
            "cv_accuracy": float(
                cv_results["mean_test_accuracy"][best_index]
            ),
            "train_recall": train_recall,
            "validation_recall": validation_recall,
            "recall_gap": train_recall - validation_recall,
        }

        fitted_models[model_name] = search.best_estimator_

        print("Best parameters:", search.best_params_)
        print("CV recall:", search.best_score_)

    # 9. Evaluate Day 3 champion on the same test customers
    day3_path = MODEL_DIR / "churn_v3.joblib"

    if day3_path.exists():

        day3_model = joblib.load(day3_path)

        results["Day 3 Champion"] = {
            "holdout_metrics": evaluate_model(
                day3_model,
                X_test_raw,
                y_test_raw,
            )
        }

    else:
        print(
            "\nWARNING: Day 3 model not found. "
            "Skipping Day 3 holdout comparison."
        )

    # 10. Evaluate Day 4 models
    for model_name, model in fitted_models.items():

        results[model_name]["holdout_metrics"] = evaluate_model(
            model,
            X_test,
            y_test,
        )

    # 11. Select champion using CV recall, then ROC-AUC
    champion_name = max(
        fitted_models,
        key=lambda name: (
            results[name]["cv_recall"],
            results[name]["cv_roc_auc"],
        ),
    )

    champion = fitted_models[champion_name]

    # 12. Save complete preprocessing + feature selection + model pipeline
    model_path = MODEL_DIR / "churn_v4.joblib"

    joblib.dump(champion, model_path)

    # 13. Save results
    output = {
        "champion": champion_name,
        "champion_model_path": str(model_path),
        "results": results,
    }

    results_path = MODEL_DIR / "day4_results.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    # 14. Display results
    print("\n========== DAY 4 RESULTS ==========")
    print("Champion:", champion_name)

    for name, result in results.items():
        print(f"\n{name}")
        print(json.dumps(result, indent=4))

    print("\nSaved model:", model_path)
    print("Saved results:", results_path)


if __name__ == "__main__":
    main()
