"""
train_crop_model.py
===================
Crop Recommendation Model Training Pipeline

This script trains, evaluates, and persists machine learning classification models
for agricultural crop recommendation using the genuine 2,200-sample Crop Recommendation dataset.

Models evaluated:
  - Random Forest Classifier (Default best model)
  - Decision Tree Classifier
  - Gaussian Naive Bayes

Output Artifacts:
  - models/crop_recommendation_model.pkl : Serialized best model & agronomic benchmarks
  - models/crop_model_metrics.json        : Unseen test evaluation metrics & confusion matrix
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = os.path.join("data", "crop_recommendation.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "crop_model_metrics.json")

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the crop recommendation dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure data/crop_recommendation.csv exists."
        )
    df = pd.read_csv(data_path)
    
    # Validation checks
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    return df


def calculate_agronomic_benchmarks(df: pd.DataFrame) -> dict:
    """
    Compute mean, min, and max nutrient and climate requirements for each crop.
    Used by the recommender for agronomic gap analysis.
    """
    benchmarks = {}
    crops = df[TARGET_COLUMN].unique()
    for crop in sorted(crops):
        crop_data = df[df[TARGET_COLUMN] == crop]
        benchmarks[str(crop)] = {}
        for feat in FEATURE_COLUMNS:
            benchmarks[str(crop)][feat] = {
                "mean": round(float(crop_data[feat].mean()), 2),
                "min": round(float(crop_data[feat].min()), 2),
                "max": round(float(crop_data[feat].max()), 2),
                "std": round(float(crop_data[feat].std()), 2)
            }
    return benchmarks


def train_and_evaluate(data_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """
    Execute full training, cross-model comparison, evaluation, and saving.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 70)
    print("  Smart Agriculture - Crop Recommendation Model Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Loading genuine dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"      Loaded {len(df)} samples across {df[TARGET_COLUMN].nunique()} crop classes.")
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    classes = sorted(y.unique().tolist())

    # 2. Train / Test Split
    print(f"\n[2/5] Splitting data into 80% train and 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training samples : {len(X_train)}")
    print(f"      Testing samples  : {len(X_test)}")

    # 3. Model Candidates
    candidates = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=random_state
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=12, random_state=random_state
        ),
        "Gaussian Naive Bayes": GaussianNB()
    }

    print("\n[3/5] Evaluating candidate algorithms on unseen test data:")
    print("-" * 70)
    print(f"{'Algorithm':<30} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weighted':<10}")
    print("-" * 70)

    best_name = None
    best_model = None
    best_acc = -1.0
    comparison_summary = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wgt = f1_score(y_test, y_pred, average="weighted")
        
        comparison_summary[name] = {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_wgt), 4)
        }
        
        print(f"{name:<30} | {acc * 100:>8.2f}% | {f1_mac:>10.4f} | {f1_wgt:>10.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print("-" * 70)
    print(f"  Best Algorithm: {best_name} with {best_acc * 100:.2f}% test accuracy!")

    # 4. Detailed Evaluation of the Best Model
    print(f"\n[4/5] Computing comprehensive test metrics for {best_name}...")
    y_pred_best = best_model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred_best))
    prec_macro = float(precision_score(y_test, y_pred_best, average="macro"))
    prec_weighted = float(precision_score(y_test, y_pred_best, average="weighted"))
    rec_macro = float(recall_score(y_test, y_pred_best, average="macro"))
    rec_weighted = float(recall_score(y_test, y_pred_best, average="weighted"))
    f1_mac = float(f1_score(y_test, y_pred_best, average="macro"))
    f1_wgt = float(f1_score(y_test, y_pred_best, average="weighted"))

    report_dict = classification_report(y_test, y_pred_best, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_best, labels=classes).tolist()

    # Feature Importances (if available)
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        for feat, imp in zip(FEATURE_COLUMNS, best_model.feature_importances_):
            feature_importances[feat] = round(float(imp), 4)
        print("\n  Feature Importances (Random Forest):")
        for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {feat:<12} : {imp * 100:>5.2f}%")

    # Agronomic benchmarks per crop
    benchmarks = calculate_agronomic_benchmarks(df)

    # 5. Save Artifacts
    print(f"\n[5/5] Saving model and evaluation metrics to disk...")
    
    # Save Model Bundle
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": FEATURE_COLUMNS,
        "target_classes": classes,
        "feature_importances": feature_importances,
        "agronomic_benchmarks": benchmarks,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wgt, 4)
        },
        "metadata": {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "dataset_rows": len(df),
            "num_crops": len(classes)
        }
    }
    
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"      [OK] Saved trained model bundle -> {MODEL_PATH}")

    # Save JSON Metrics for UI & Reporting
    metrics_json = {
        "best_model": best_name,
        "training_date": model_bundle["metadata"]["training_date"],
        "dataset_rows": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "overall_metrics": model_bundle["metrics"],
        "model_comparison": comparison_summary,
        "feature_importances": feature_importances,
        "classes": classes,
        "confusion_matrix": cm,
        "classification_report": {
            k: v for k, v in report_dict.items() if k not in ["accuracy"]
        }
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"      [OK] Saved evaluation metrics  -> {METRICS_PATH}")

    print("\n" + "=" * 70)
    print(f"  Training complete! Overall Test Accuracy: {acc * 100:.2f}%")
    print("=" * 70)
    return model_bundle


if __name__ == "__main__":
    train_and_evaluate()
