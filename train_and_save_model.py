# train_and_save_model.py
import pandas as pd
import numpy as np
import pickle
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.api.types import is_numeric_dtype

# ----- load data -----
df = pd.read_csv('car data.csv')
print("Columns in CSV:", list(df.columns))

# ----- detect target column (common names) -----
possible_targets = ['Selling_Price','selling_price','Selling_price','Price','price','Resale_Price','Resale price','target']
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    # try to infer a numeric column that's likely the price (best-effort)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # remove likely feature columns
    for col in ['Kms_Driven','Year','Owner','Mileage','Power','Engine','Seats']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if numeric_cols:
        target_col = numeric_cols[0]
        print(f"No common target name found — using inferred numeric column: {target_col}")
    else:
        print("Couldn't detect target column. Edit the script and set the correct target column name.")
        sys.exit(1)

print("Using target column:", target_col)

# ----- select features we expect (adaptable) -----
expected = ['Car_Name','Year','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']
features = [c for c in expected if c in df.columns]
print("Using features:", features)

# ----- create Age from Year if Year present -----
if 'Year' in df.columns:
    current_year = datetime.now().year
    df['Age'] = current_year - df['Year']
    # remove Year from features and use Age instead
    features = [f for f in features if f != 'Year']
    if 'Age' not in features:
        features.append('Age')

# ----- Prepare X, y -----
X = df[features].copy()
y = df[target_col].copy()

# drop rows with NA in X or y
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

# detect numeric vs categorical for ColumnTransformer
numeric_features = [c for c in X.columns if is_numeric_dtype(X[c])]
categorical_features = [c for c in X.columns if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ----- build preprocessing + model pipeline -----
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ----- train/test split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- fit -----
print("Training model...")
model.fit(X_train, y_train)

# ----- evaluate -----
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
import numpy as np
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))


# ----- save model & feature list -----
pickle.dump(model, open('car_price_model.pkl', 'wb'))
pickle.dump(list(X.columns), open('feature_columns.pkl', 'wb'))
print("Saved 'car_price_model.pkl' and 'feature_columns.pkl' in project folder.")
