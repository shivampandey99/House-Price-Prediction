import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("Model/linear_regression_model.pkl")
    scaler = joblib.load("Model/scaler.pkl")
    return model, scaler


model, scaler = load_model()
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏠 House Price Predictor")

    st.write(
        "This application predicts house prices using a "
        "Linear Regression model trained on the USA Housing Dataset."
    )

    st.divider()

    st.subheader("Model Information")

    st.markdown("**Algorithm:** Linear Regression")
    st.markdown("**Dataset:** USA Housing")
    st.markdown("**Features:** 6")
    st.markdown("**R² Score:** 0.918")


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter the area and housing information below. "
    "The trained machine-learning model will estimate the house price."
)

st.divider()

st.header("Property Information")


# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2 = st.columns(2)


# -------------------- COLUMN 1 --------------------

with col1:

    avg_income = st.number_input(
        "Average Area Income",
        min_value=17796.0,
        max_value=107702.0,
        value=None,
        placeholder="e.g. 65000",
        step=1000.0
    )

    avg_rooms = st.number_input(
        "Average Area Number of Rooms",
        min_value=3.0,
        max_value=11.0,
        value=None,
        placeholder="e.g. 7",
        step=0.1
    )


# -------------------- COLUMN 2 --------------------

with col2:

    house_age = st.number_input(
        "Average Area House Age",
        min_value=2.0,
        max_value=10.0,
        value=None,
        placeholder="e.g. 5.5",
        step=0.1
    )

    avg_bedrooms = st.number_input(
        "Average Area Number of Bedrooms",
        min_value=2.0,
        max_value=7.0,
        value=None,
        placeholder="e.g. 4",
        step=0.1
    )


# -------------------- FULL WIDTH INPUT --------------------

population = st.number_input(
    "Area Population",
    min_value=172.0,
    max_value=69622.0,
    value=None,
    placeholder="e.g. 35000",
    step=1000.0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True
):

    # Check whether every field has been filled
    if (
        avg_income is None
        or house_age is None
        or avg_rooms is None
        or avg_bedrooms is None
        or population is None
    ):

        st.warning(
            "Please enter values in all property information fields."
        )

    else:

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        # This is the same engineered feature used during
        # model training.
        rooms_per_bedroom = avg_rooms / avg_bedrooms


        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                avg_income,
                house_age,
                avg_rooms,
                avg_bedrooms,
                population,
                rooms_per_bedroom
            ]],
            columns=[
                "avg_area_income",
                "avg_area_house_age",
                "avg_area_num_rooms",
                "avg_area_num_bedrooms",
                "area_population",
                "rooms_per_bedroom"
            ]
        )


        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        input_scaled = scaler.transform(input_data)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_scaled)[0]


        # ----------------------------------------------------
        # POPULATION CATEGORY
        # ----------------------------------------------------

        # Categories based on the population ranges identified
        # during data analysis.

        if population <= 31800.302253:
            population_category = "Low"

        elif population <= 40456.013364:
            population_category = "Medium"

        else:
            population_category = "High"


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("Prediction Result")

        st.success(
            f"Estimated House Price: ${prediction:,.2f}"
        )

        st.info(
            f"Population Category: {population_category}"
        )