# scripts/model_training.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "./data/processed/cleaned_data.csv"
MODEL_PATH = "./models"

def load_data(file_path=DATA_PATH, target_col="proxy_default_label"):
    """
    Loads processed data and ensures the target column exists.
    """
    df = pd.read_csv(file_path)

    # Ensure target column exists
    if target_col not in df.columns:
        raise KeyError(
            f"❌ Target column '{target_col}' not found! Available columns: {df.columns.tolist()}"
        )
    
    return df

def prepare_data(df, target_col="proxy_default_label"):
    """
    Splits data into features (X) and target (y).
    """
    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude non-numeric columns from features, except the target column
    feature_cols = [col for col in df.columns if col not in [target_col, "CustomerId"] + non_numeric_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model and prints metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}\n")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc
    }

def main():
    # 1. Load Data
    df = load_data()

    # 2. Split Data
    X, y, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train Logistic Regression
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    logreg_metrics = evaluate_model(logreg, X_test, y_test, model_name="Logistic Regression")

    # 4. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, model_name="Random Forest")

    # 5. Compare & Select Best Model
    best_model, model_name = (rf, "RandomForest") if rf_metrics["roc_auc"] > logreg_metrics["roc_auc"] else (logreg, "LogisticRegression")
    
    print(f"\n✅ Best Model Selected: {model_name}")

    # 6. Save the Best Model
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_filepath = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    joblib.dump(best_model, model_filepath)
    print(f"\n✅ Model saved to {model_filepath}")

if __name__ == "__main__":
    main()
