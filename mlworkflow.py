"""
MLflow Experiment Tracking — Airbnb Price Prediction
With Hyperparameter Tuning using GridSearchCV for each model.

Each model runs in TWO MLflow runs:
  1. Baseline  — default params (for comparison)
  2. Tuned     — best params found by GridSearchCV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import Ridge          # Ridge = LinearReg + regularisation tuning
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────

DATA_PATH = "D:\\expernetic_internship\\datasets\\listings_cleaned.csv"

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=[
    'id', 'name', 'host_id', 'host_name',
    'last_review', 'license',
    'number_of_reviews_ltm', 'Unnamed: 0'
], errors='ignore')

df['log_price'] = np.log1p(df['price'])

cat_cols = ['neighbourhood', 'room_type', 'Rental_stratergy', 'host_business_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['price', 'log_price'])
y = df_encoded['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print("─" * 60)


# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_metrics(y_true, y_pred):
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2"  : r2_score(y_true, y_pred),
    }

def log_residual_plot(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted log_price')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} — Residuals vs Predicted')
    axes[1].hist(residuals, bins=50, edgecolor='black', color='steelblue')
    axes[1].set_title(f'{model_name} — Residual Distribution')
    axes[1].set_xlabel('Residual')
    plt.tight_layout()
    path = f"residuals_{model_name}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path)

def log_actual_vs_fitted(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual log_price')
    ax.set_ylabel('Predicted log_price')
    ax.set_title(f'{model_name} — Actual vs Fitted')
    ax.grid(True, linestyle='--', alpha=0.5)
    path = f"actual_vs_fitted_{model_name}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path)

def log_to_mlflow(run_name, model, params, y_test, y_pred, log_fn, log_key, tag):
    """Logs one full MLflow run."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("tuning", tag)                    # "baseline" or "tuned"

        test_m = get_metrics(y_test, y_pred)

        mlflow.log_params(params)

        mlflow.log_metric("test_MAE",  test_m["MAE"])
        mlflow.log_metric("test_RMSE", test_m["RMSE"])
        mlflow.log_metric("test_R2",   test_m["R2"])

        log_residual_plot(y_test, y_pred, run_name)
        log_actual_vs_fitted(y_test, y_pred, run_name)
        log_fn(model, log_key)

        print(f"   Test → MAE: {test_m['MAE']:.4f} | RMSE: {test_m['RMSE']:.4f} | R²: {test_m['R2']:.4f}")
        print(f"   ✅ Logged ({tag})\n")


# ─────────────────────────────────────────────
# 3. SET EXPERIMENT
# ─────────────────────────────────────────────

import dagshub
dagshub.init(repo_owner="oshinijayathunga",
             repo_name="Airbnb-price-prediction",
             mlflow=True)

mlflow.set_experiment("Airbnb_Price_Prediction_Tuned")

# ═════════════════════════════════════════════
# MODEL 1 — RIDGE REGRESSION
# (Linear Regression + alpha regularisation)
# ═════════════════════════════════════════════

print("\n" + "═"*60)
print("  MODEL 1: Ridge Regression")
print("═"*60)

# ── Baseline ──
print("  [1/2] Baseline ...")
baseline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
baseline_ridge.fit(X_train, y_train)
y_pred_ridge_base = baseline_ridge.predict(X_test)

log_to_mlflow(
    run_name = "Ridge_Regression_Baseline",
    model    = baseline_ridge,
    params   = {"model_type": "Ridge", "alpha": 1.0, "tuned": False},
    y_test   = y_test,
    y_pred   = y_pred_ridge_base,
    log_fn   = mlflow.sklearn.log_model,
    log_key  = "ridge_baseline",
    tag      = "baseline"
)

# ── Tuned with GridSearchCV ──
print("  [2/2] Grid search tuning ...")
ridge_param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
}
ridge_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    param_grid = ridge_param_grid,
    cv         = 5,
    scoring    = 'neg_mean_absolute_error',
    n_jobs     = -1,
    verbose    = 0
)
ridge_grid.fit(X_train, y_train)
best_ridge  = ridge_grid.best_estimator_
y_pred_ridge_tuned = best_ridge.predict(X_test)
best_alpha  = ridge_grid.best_params_['model__alpha']
print(f"   Best alpha: {best_alpha}")

log_to_mlflow(
    run_name = "Ridge_Regression_Tuned",
    model    = best_ridge,
    params   = {"model_type": "Ridge", "alpha": best_alpha, "tuned": True},
    y_test   = y_test,
    y_pred   = y_pred_ridge_tuned,
    log_fn   = mlflow.sklearn.log_model,
    log_key  = "ridge_tuned",
    tag      = "tuned"
)


# ═════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST
# ═════════════════════════════════════════════

print("═"*60)
print("  MODEL 2: Random Forest")
print("═"*60)

# ── Baseline ──
print("  [1/2] Baseline ...")
baseline_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
baseline_rf.fit(X_train, y_train)
y_pred_rf_base = baseline_rf.predict(X_test)

log_to_mlflow(
    run_name = "Random_Forest_Baseline",
    model    = baseline_rf,
    params   = {"model_type": "RandomForest", "n_estimators": 100,
                "max_depth": "None", "min_samples_split": 2,
                "max_features": "1.0", "tuned": False},
    y_test   = y_test,
    y_pred   = y_pred_rf_base,
    log_fn   = mlflow.sklearn.log_model,
    log_key  = "rf_baseline",
    tag      = "baseline"
)

# ── Tuned with GridSearchCV ──
print("  [2/2] Grid search tuning (this takes ~5 min) ...")
rf_param_grid = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features'    : ['sqrt', 'log2', 0.5],
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid = rf_param_grid,
    cv         = 3,              # 3-fold to keep time reasonable
    scoring    = 'neg_mean_absolute_error',
    n_jobs     = -1,
    verbose    = 1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
print(f"   Best params: {rf_grid.best_params_}")

log_to_mlflow(
    run_name = "Random_Forest_Tuned",
    model    = best_rf,
    params   = {
        "model_type"       : "RandomForest",
        "tuned"            : True,
        **{k: str(v) for k, v in rf_grid.best_params_.items()}
    },
    y_test   = y_test,
    y_pred   = y_pred_rf_tuned,
    log_fn   = mlflow.sklearn.log_model,
    log_key  = "rf_tuned",
    tag      = "tuned"
)


# ═════════════════════════════════════════════
# MODEL 3 — XGBOOST
# ═════════════════════════════════════════════

print("═"*60)
print("  MODEL 3: XGBoost")
print("═"*60)

# ── Baseline ──
print("  [1/2] Baseline ...")
baseline_xgb = XGBRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=6, random_state=42, n_jobs=-1, verbosity=0
)
baseline_xgb.fit(X_train, y_train)
y_pred_xgb_base = baseline_xgb.predict(X_test)

log_to_mlflow(
    run_name = "XGBoost_Baseline",
    model    = baseline_xgb,
    params   = {"model_type": "XGBoost", "n_estimators": 200,
                "learning_rate": 0.05, "max_depth": 6, "tuned": False},
    y_test   = y_test,
    y_pred   = y_pred_xgb_base,
    log_fn   = mlflow.xgboost.log_model,
    log_key  = "xgb_baseline",
    tag      = "baseline"
)

# ── Tuned with GridSearchCV ──
print("  [2/2] Grid search tuning (this takes ~5 min) ...")
xgb_param_grid = {
    'n_estimators'  : [100, 200, 300],
    'learning_rate' : [0.01, 0.05, 0.1, 0.2],
    'max_depth'     : [3, 5, 6, 8],
    'subsample'     : [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_grid = xgb_param_grid,
    cv         = 3,
    scoring    = 'neg_mean_absolute_error',
    n_jobs     = -1,
    verbose    = 1
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test)
print(f"   Best params: {xgb_grid.best_params_}")

log_to_mlflow(
    run_name = "XGBoost_Tuned",
    model    = best_xgb,
    params   = {
        "model_type": "XGBoost",
        "tuned"     : True,
        **{k: str(v) for k, v in xgb_grid.best_params_.items()}
    },
    y_test   = y_test,
    y_pred   = y_pred_xgb_tuned,
    log_fn   = mlflow.xgboost.log_model,
    log_key  = "xgb_tuned",
    tag      = "tuned"
)


# ─────────────────────────────────────────────
# 4. FINAL COMPARISON SUMMARY
# ─────────────────────────────────────────────

print("\n" + "═"*60)
print("  FINAL COMPARISON SUMMARY")
print("═"*60)

results = {
    "Ridge Baseline" : get_metrics(y_test, y_pred_ridge_base),
    "Ridge Tuned"    : get_metrics(y_test, y_pred_ridge_tuned),
    "RF Baseline"    : get_metrics(y_test, y_pred_rf_base),
    "RF Tuned"       : get_metrics(y_test, y_pred_rf_tuned),
    "XGBoost Baseline": get_metrics(y_test, y_pred_xgb_base),
    "XGBoost Tuned"  : get_metrics(y_test, y_pred_xgb_tuned),
}

print(f"\n  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}")
for name, m in results.items():
    print(f"  {name:<22} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}")

print(f"\n  ✅ All 6 runs logged to MLflow")
print(f"  Run: python -m mlflow ui  →  http://localhost:5000")
print("═"*60)