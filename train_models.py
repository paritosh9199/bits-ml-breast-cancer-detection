"""Train the five required classifiers and create reproducible app artifacts."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from model.modeling import build_models, evaluate_classifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "artifacts"


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target.rename("target")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in build_models(features.columns.tolist()).items():
        model.fit(X_train, y_train)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")
        rows.append({"ML Model Name": name, **evaluate_classifier(model, X_test, y_test)})

    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        ROOT / "test_data.csv", index=False
    )
    pd.DataFrame(rows).to_csv(MODEL_DIR / "baseline_metrics.csv", index=False)
    joblib.dump(
        {
            "feature_names": features.columns.tolist(),
            "target_name": "target",
            "target_labels": {0: "malignant", 1: "benign"},
            "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        },
        MODEL_DIR / "metadata.joblib",
    )
    print(f"Saved {len(rows)} models and {len(X_test)} test rows.")


if __name__ == "__main__":
    main()

