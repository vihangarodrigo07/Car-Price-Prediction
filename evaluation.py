# evaluation.ipynb

# --- Cell 1: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Cell 2: Load dataset ---
df = pd.read_csv("car data.csv")
df.head()

# --- Cell 3: Basic info ---
df.info()
df.describe()

# --- Cell 4: Preprocessing ---
# Convert categorical to dummies
df_processed = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_processed.drop("Selling_Price", axis=1)
y = df_processed["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Cell 5: Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Cell 6: Predictions ---
y_pred = model.predict(X_test)

# --- Cell 7: Evaluation metrics ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# --- Cell 8: Predicted vs Actual plot ---
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True, alpha=0.3)
plt.savefig("results/actual_vs_pred.png")
plt.show()

# --- Cell 9: Residuals plot ---
residuals = y_test - y_pred
plt.figure(figsize=(7,5))
sns.histplot(residuals, bins=20, kde=True, color="red")
plt.xlabel("Residual (Error)")
plt.title("Distribution of Prediction Errors")
plt.grid(True, alpha=0.3)
plt.savefig("results/residuals.png")
plt.show()

# --- Cell 10: Feature Importance (coefficients) ---
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df)
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.grid(True, alpha=0.3)
plt.savefig("results/feature_importance.png")
plt.show()

coef_df
