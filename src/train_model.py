"""
train_model.py
----------------
Trains and compares multiple regression models for student marks
prediction, tunes the Gradient Boosting Regressor with GridSearchCV,
and saves the best model + preprocessor + evaluation plots.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import Preprocessor, load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "kaggle_students_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name:22s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R2: {r2:.4f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def main():
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    pre = Preprocessor()
    X, y = pre.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    print("\nTraining baseline models...\n")

    lr = LinearRegression().fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lr, X_test, y_test))

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    results.append(evaluate("Ridge Regression", ridge, X_test, y_test))

    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    gbr_default = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    results.append(evaluate("Gradient Boosting (default)", gbr_default, X_test, y_test))

    print("\nTuning Gradient Boosting Regressor with GridSearchCV...\n")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    best_gbr = grid_search.best_estimator_
    result = evaluate("Gradient Boosting (tuned)", best_gbr, X_test, y_test)
    results.append(result)

    # Pick the best model overall by R2
    best_overall = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best_overall['name']} (R2 = {best_overall['r2']:.4f})")

    model_map = {
        "Linear Regression": lr,
        "Ridge Regression": ridge,
        "Random Forest": rf,
        "Gradient Boosting (default)": gbr_default,
        "Gradient Boosting (tuned)": best_gbr,
    }
    best_model = model_map[best_overall["name"]]

    # Save model + preprocessor
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)

    # Save a residual std for confidence-interval estimation at inference time
    residuals = y_test - best_model.predict(X_test)
    resid_std = float(np.std(residuals))
    with open(os.path.join(MODEL_DIR, "residual_std.pkl"), "wb") as f:
        pickle.dump(resid_std, f)

    print(f"\nSaved best model, preprocessor, and residual std to '{MODEL_DIR}/'")

    # ---- Plots ----
    # 1. Model comparison bar chart
    plt.figure(figsize=(8, 5))
    names = [r["name"] for r in results]
    r2s = [r["r2"] for r in results]
    plt.barh(names, r2s, color="#4C72B0")
    plt.xlabel("R2 Score")
    plt.title("Model Comparison (R2 Score)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # 2. Actual vs Predicted for best model
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_overall["preds"], alpha=0.5, color="#55A868")
    lims = [min(y_test.min(), best_overall["preds"].min()), max(y_test.max(), best_overall["preds"].max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title(f"Actual vs Predicted - {best_overall['name']}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # 3. Feature importance: tree-based models expose feature_importances_,
    # linear models expose coef_ (magnitude used as a proxy for importance
    # since features are standardized).
    feat_names = pre.feature_names_
    importances = None
    importance_label = "Importance"

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_)
        importance_label = "|Coefficient| (standardized features)"

    if importances is not None:
        order = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feat_names)[order], importances[order], color="#C44E52")
        plt.xlabel(importance_label)
        plt.title(f"Feature Importance - {best_overall['name']}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    print(f"Plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
