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
    "Enter your property details to get an estimated house price "
    "using a Linear Regression model."

    )

    st.divider()

    st.subheader("Model Information")

    st.markdown("**Algorithm:** Linear Regression")
    st.markdown("**Dataset:** USA Housing")
    st.markdown("**Features:** 4")
    st.markdown("**R² Score:** 0.9181")


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    "<h1 style='text-align: center;'>🏠 House Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>"
    "Enter your property details below to get an estimated house price."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

st.header("Property Information")

# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2 = st.columns(2)


# -------------------- COLUMN 1 --------------------

with col1:

    income = st.number_input(
        "Annual Income ($)",
        min_value=17796,
        max_value=107702,
        value=None,
        placeholder="e.g. 65000",
        step=1000
    )

    rooms = st.number_input(
        "Number of Rooms",
        min_value=3,
        max_value=11,
        value=None,
        placeholder="e.g. 6",
        step=1
    )


# -------------------- COLUMN 2 --------------------

with col2:

    house_age = st.number_input(
        "House Age (Years)",
        min_value=2,
        max_value=10,
        value=None,
        placeholder="e.g. 5",
        step=1
    )

    population = st.number_input(
        "Area Population",
        min_value=172,
        max_value=69622,
        value=None,
        placeholder="e.g. 35000",
        step=1000
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
        income is None
        or house_age is None
        or rooms is None
        or population is None
    ):
        st.warning(
            "Please enter values in all property information fields."
        )

    else:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                income,
                house_age,
                rooms,
                population
            ]],
            columns=[
                "income",
                "house_age",
                "rooms",
                "population"
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