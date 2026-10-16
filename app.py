import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail Demand Predictor",
    page_icon="📦",
    layout="centered",
)

MODEL_PATH = "retail_demand_model.pkl"
DATA_PATH = "sales_data.csv"

# Check if required files exist (They exist locally, but not on GitHub)
HAS_MODEL = os.path.exists(MODEL_PATH)
HAS_DATA = os.path.exists(DATA_PATH)

@st.cache_resource
def load_model():
    if HAS_MODEL:
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_reference_data():
    if HAS_DATA:
        return pd.read_csv(DATA_PATH)
    
    # Fallback mock data for lookup tables if CSV is ignored on GitHub
    mock_data = {
        "Product ID": [f"P{i:03d}" for i in range(1, 11)],
        "Category": ["Electronics", "Clothing", "Home", "Electronics", "Clothing", "Home", "Groceries", "Groceries", "Toys", "Toys"],
        "Store ID": [f"S00{i}" for i in range(1, 6)] * 2,
        "Region": ["North", "South", "East", "West", "Central"] * 2,
        "Weather Condition": ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Sunny", "Rainy", "Cloudy", "Snowy", "Windy"],
        "Seasonality": ["Spring", "Summer", "Autumn", "Winter", "Holiday", "Spring", "Summer", "Autumn", "Winter", "Holiday"]
    }
    return pd.DataFrame(mock_data)

model = load_model()
df = load_reference_data()

st.title("📦 Retail Demand Predictor")
st.write("Estimate product demand to support inventory planning and replenishment.")

# Alert user about execution environment mode
if not HAS_MODEL or not HAS_DATA:
    st.info(
        "💡 **Running in Online Demo Mode**\n\n"
        "Because the 1GB model and data files are ignored in the GitHub repository, "
        "this web app uses rule-based predictive fallback calculations to demonstrate functionality."
    )
else:
    st.info(
        "This application uses the trained Random Forest Regression model from the project. "
        "Some values such as Category and Region are filled automatically from the selected product and store."
    )

# Build reliable lookup tables from the actual project data.
product_lookup = (
    df[["Product ID", "Category"]]
    .drop_duplicates()
    .sort_values("Product ID")
)
product_lookup = product_lookup.groupby("Product ID", as_index=False).first()

store_lookup = (
    df[["Store ID", "Region"]]
    .drop_duplicates()
    .sort_values("Store ID")
)
store_lookup = store_lookup.groupby("Store ID", as_index=False).first()

st.header("1. Basic Information")
col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Prediction Date")
    product_id = st.selectbox("Product", product_lookup["Product ID"].tolist())

with col2:
    store_id = st.selectbox("Store", store_lookup["Store ID"].tolist())

# Corrected .iloc syntax error
selected_category = product_lookup.loc[
    product_lookup["Product ID"] == product_id, "Category"
].values[0]
selected_region = store_lookup.loc[
    store_lookup["Store ID"] == store_id, "Region"
].values[0]

c1, c2 = st.columns(2)
with c1:
    st.text_input("Category", value=selected_category, disabled=True)
with c2:
    st.text_input("Region", value=selected_region, disabled=True)

st.header("2. Sales and Inventory")
col1, col2 = st.columns(2)

with col1:
    inventory_level = st.number_input(
        "Current Inventory (units)", min_value=0, value=100, step=1
    )
    units_sold = st.number_input(
        "Recent Units Sold", min_value=0, value=50, step=1,
        help="Historical sales value used by the trained model."
    )
    units_ordered = st.number_input(
        "Units Ordered", min_value=0, value=100, step=1
    )

with col2:
    price = st.number_input("Product Price", min_value=0.0, value=50.0, step=0.01)
    discount = st.number_input("Discount (%)", min_value=0.0, value=0.0, step=1.0)
    competitor_pricing = st.number_input(
        "Competitor Price", min_value=0.0, value=50.0, step=0.01
    )

st.header("3. Market Conditions")
col1, col2 = st.columns(2)

with col1:
    weather = st.selectbox(
        "Weather Condition",
        sorted(df["Weather Condition"].dropna().unique().tolist())
    )
    promotion = st.selectbox(
        "Promotion",
,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    seasonality = st.selectbox(
        "Seasonality",
        sorted(df["Seasonality"].dropna().unique().tolist())
    )
    epidemic = st.selectbox(
        "Epidemic Condition",
,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.caption("The app automatically creates Year, Month, Day, and Day of Week from the selected date.")

if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
    if HAS_MODEL and model is not None:
        # --- TRUE ML MODEL RUN (Runs on your local computer) ---
        input_df = pd.DataFrame([{
            "Date": pd.Timestamp(date),
            "Store ID": store_id,
            "Product ID": product_id,
            "Category": selected_category,
            "Region": selected_region,
            "Inventory Level": inventory_level,
            "Units Sold": units_sold,
            "Units Ordered": units_ordered,
            "Price": price,
            "Discount": discount,
            "Weather Condition": weather,
            "Promotion": promotion,
            "Competitor Pricing": competitor_pricing,
            "Seasonality": seasonality,
            "Epidemic": epidemic,
        }])

        input_df["Date"] = pd.to_datetime(input_df["Date"])
        input_df["Year"] = input_df["Date"].dt.year
        input_df["Month"] = input_df["Date"].dt.month
        input_df["Day"] = input_df["Date"].dt.day
        input_df["DayOfWeek"] = input_df["Date"].dt.dayofweek
        input_df = input_df.drop(columns=["Date"])

        prediction = float(model.predict(input_df)[0])
        predicted_demand = max(0, round(prediction))
    else:
        # --- DEMO FALLBACK LOGIC (Runs on Streamlit Community Cloud) ---
        base_demand = units_sold * 1.15
        if promotion == 1:
            base_demand *= 1.25
        if discount > 0:
            base_demand *= (1 + (discount / 100))
        if competitor_pricing < price:
            base_demand *= 0.85
        if weather in ["Rainy", "Snowy"] and selected_category != "Groceries":
            base_demand *= 0.75
            
        predicted_demand = max(5, round(base_demand))

    difference = predicted_demand - inventory_level
    st.success("Prediction completed!")

    st.header("4. Demand Result")
    st.caption("Use this result as decision-support information for inventory planning.")

    # Main result cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Demand", f"{predicted_demand} units")
    with c2:
        st.metric("Current Inventory", f"{inventory_level} units")
    with c3:
        if difference > 0:
            st.metric("Inventory Difference", f"-{difference} units")
        else:
            st.metric("Inventory Difference", f"+{abs(difference)} units")

    # Clear, human-readable status and recommendation
    if difference > 0:
        st.error("🔴 REPLENISH STOCK")
        st.subheader("What does this mean?")
        st.write(
            f"The model predicts a demand of *{predicted_demand} units*, but the current "
            f"inventory is only *{inventory_level} units*. The inventory is below the predicted demand."
        )
        st.info(
            f"**Suggested replenishment: {difference} units**\n\n"
            "This is an estimate to support inventory planning, not an automatic purchase order."
        )
    else:
        st.success("🟢 STOCK SUFFICIENT")
        st.subheader("What does this mean?")
        st.write(
            f"The model predicts a demand of *{predicted_demand} units*, while the current "
            f"inventory is *{inventory_level} units*. The current inventory is enough to cover "
            "the predicted demand based on this model prediction."
        )
        st.info(
            f"**Inventory above predicted demand: {abs(difference)} units**\n\n"
            "Continue monitoring inventory and actual sales conditions."
        )

    st.caption(
        "Important: The prediction is an estimate from the trained Random Forest model. "
        "Actual inventory decisions should also consider supplier lead times, current sales, "
        "and other business conditions."
    )

st.divider()
st.caption(
    "Project model: Random Forest Regression | 200 trees | R² = 0.8711 | "
    "Academic decision-support prototype"
)
st.caption(
    "Note: Recent Units Sold is included because it is part of the trained model. "
    "For future-demand deployment, confirm that this value is available before the prediction period."
)
