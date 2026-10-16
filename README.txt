
# Retail Inventory Demand Predictor

## 1. Install Python
Install Python 3.10+ on Windows and make sure Python is added to PATH.

## 2. Open Command Prompt in this folder

Example:
cd C:\path\to\retail_demand_app

## 3. Install packages

pip install -r requirements.txt

## 4. Train the model (optional)

The trained model is already included as `retail_demand_model.pkl`.

If you want to retrain it from the CSV:

python train_model.py

The script uses:
- chronological 80/20 split
- date feature extraction
- OneHotEncoder for categorical variables
- Random Forest Regression
- n_estimators=200
- random_state=42
- max_features="sqrt"

## 5. Run the application

streamlit run app.py

Your browser should open the application.

## 6. Stop the application

Press Ctrl+C in Command Prompt.

## Important academic note

This application reproduces the model documented in the project. The documentation reports Random Forest performance of approximately:

MAE = 12.1199
MSE = 250.4647
RMSE = 15.8261
R² = 0.8711

Before claiming that the application is a real-world future forecasting system, verify that every predictor is genuinely available before the demand period. In particular, investigate whether `Units Sold` could introduce target leakage.
