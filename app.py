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

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    return df

model = load_model()
df = load_reference_data()

st.title("📦 Retail Demand Predictor")
st.write("Estimate product demand to support inventory planning and replenishment.")

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

selected_category = product_lookup.loc[
    product_lookup["Product ID"] == product_id, "Category"
].iloc[0]
selected_region = store_lookup.loc[
    store_lookup["Store ID"] == store_id, "Region"
].iloc[0]

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
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    seasonality = st.selectbox(
        "Seasonality",
        sorted(df["Seasonality"].dropna().unique().tolist())
    )
    epidemic = st.selectbox(
        "Epidemic Condition",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.caption("The app automatically creates Year, Month, Day, and Day of Week from the selected date.")

if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
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

    # Match the exact feature engineering used during model training.
    input_df["Date"] = pd.to_datetime(input_df["Date"])
    input_df["Year"] = input_df["Date"].dt.year
    input_df["Month"] = input_df["Date"].dt.month
    input_df["Day"] = input_df["Date"].dt.day
    input_df["DayOfWeek"] = input_df["Date"].dt.dayofweek
    input_df = input_df.drop(columns=["Date"])

    prediction = float(model.predict(input_df)[0])
    predicted_demand = max(0, round(prediction))
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