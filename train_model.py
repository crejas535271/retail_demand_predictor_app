
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "sales_data.csv"
MODEL_PATH = "retail_demand_model.pkl"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Prepare date and sort chronologically
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 3. Feature engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# 4. Separate target and predictors
X = df.drop(columns=["Demand", "Date"])
y = df["Demand"]

# 5. Chronological 80/20 split
split = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

def evaluate(name, estimator):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    return pipe

# 7. Baseline
linear_model = evaluate("Linear Regression", LinearRegression())

# 8. Final model
rf_model = evaluate(
    "Random Forest Regression",
    RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
)

# 9. Save final model
joblib.dump(rf_model, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")
