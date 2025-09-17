# train_and_save_model.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Config ----
DATA_PATH = "car data.csv"   # change if different
MODEL_PATH = "car_price_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
IMAGES_DIR = "images"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15  # if you want train/valid/test split manually
CURRENT_YEAR = pd.to_datetime("today").year

os.makedirs(IMAGES_DIR, exist_ok=True)

# ---- Helper metrics ----
def mape(y_true, y_pred):
    # avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

def rmsle(y_true, y_pred):
    # Root Mean Squared Logarithmic Error
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# ---- Load data ----
print("Loading data from", DATA_PATH)
df = pd.read_csv(DATA_PATH)

print("Initial rows:", df.shape[0], "columns:", df.shape[1])

# ---- Basic cleaning (adapt if your dataset differs) ----
# Guess target column names; rename if needed:
target_candidates = ['selling_price', 'Selling_Price', 'price', 'Price']
target_col = None
for t in target_candidates:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    raise RuntimeError("Can't find target column (selling price). Edit the script to use the correct column name.")

# Drop rows with missing target
df = df.dropna(subset=[target_col]).copy()

# ---- Feature engineering (the 'equations' your teammate meant) ----
# 1) age
if 'year' in df.columns:
    df['age'] = CURRENT_YEAR - df['year']
else:
    print("Warning: 'year' column not found. Add age manually if available.")

# 2) km_per_year
if 'km_driven' in df.columns and 'age' in df.columns:
    df['km_per_year'] = df['km_driven'] / df['age'].replace(0, 1)
else:
    print("Warning: 'km_driven' or 'age' missing; skipping km_per_year.")

# 3) price_per_km (useful for EDA or as feature, careful with leakage if price used)
if 'km_driven' in df.columns:
    df['price_per_km'] = df[target_col] / df['km_driven'].replace(0, 1)

# Optional: log transform target (we'll keep the main target as-is)
df = df[df[target_col] > 0]  # remove non-positive prices if any

# ---- Select features ----
# Heuristic: take numeric and some categorical columns commonly present
numeric_cols = []
for c in ['km_driven', 'age', 'km_per_year', 'price_per_km']:
    if c in df.columns:
        numeric_cols.append(c)

# pick a few categorical cols if present
categorical_cols = []
for c in ['fuel', 'seller_type', 'transmission', 'owner', 'name', 'brand', 'model']:
    if c in df.columns:
        categorical_cols.append(c)

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# Build X, y
X = df[numeric_cols + categorical_cols].copy()
y = df[target_col].values

# Fill missing numeric values (simple strategy)
for n in numeric_cols:
    X[n] = X[n].fillna(X[n].median())

# Fill missing categorical values
for c in categorical_cols:
    X[c] = X[c].fillna("Unknown")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='drop')

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
])

# Fit
print("Training model...")
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mape_val = mape(y_test, y_pred)
rmsle_val = rmsle(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.4f}")
print(f"MAPE: {mape_val:.2f}%")
print(f"RMSLE: {rmsle_val:.4f}")

# Save model & feature info
joblib.dump(model, MODEL_PATH)
joblib.dump({'numeric': numeric_cols, 'categorical': categorical_cols}, FEATURES_PATH)
print("Saved model to", MODEL_PATH)
print("Saved feature columns to", FEATURES_PATH)

# ---- Plots for report ----
# 1) Residuals histogram
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50)
plt.title("Residuals distribution (y_test - y_pred)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "residuals_hist.png"))
plt.close()

# 2) Predicted vs Actual scatter
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pred_vs_actual.png"))
plt.close()

# 3) Feature importance (if RandomForest)
try:
    rf = model.named_steps['regressor']
    # We need to get feature names after preprocessing
    # numeric feature names:
    num_names = numeric_cols
    # categorical names from OneHot
    cat_names = []
    if categorical_cols:
        ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = num_names + cat_names
    importances = rf.feature_importances_
    # plot top 20
    idxs = np.argsort(importances)[::-1][:20]
    top_names = [all_feature_names[i] for i in idxs]
    top_importances = importances[idxs]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(top_importances))[::-1], top_importances[::-1])
    plt.yticks(range(len(top_names)), top_names[::-1])
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"))
    plt.close()
except Exception as e:
    print("Could not save feature importance plot:", e)

print("Saved images to:", IMAGES_DIR)
