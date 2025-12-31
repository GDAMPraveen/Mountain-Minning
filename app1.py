import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Mountain Mining Approval System",
    page_icon="⛏️",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("mining_model.pkl")

model = load_model()

# Sidebar Navigation
st.sidebar.title("⛏️ Mining System")
page = st.sidebar.radio("Navigation", ["Input Details", "Result"])

# Session State
if "input_data" not in st.session_state:
    st.session_state["input_data"] = None

# PAGE 1: INPUT PAGE
if page == "Input Details":
    st.title("📝 Mining Site Input Details")

    st.markdown("Enter geographical, environmental and risk parameters.")

    with st.form("mining_form"):
        elevation = st.number_input("Elevation (meters)", 100, 5000, 1200)
        slope = st.slider("Slope (degrees)", 0, 60, 20)
        forest_cover = st.slider("Forest Cover (%)", 0, 100, 40)

        protected = st.selectbox("Protected Area", [0, 1])
        wildlife = st.selectbox("Wildlife Sanctuary", [0, 1])

        rainfall = st.number_input("Annual Rainfall (mm)", 500, 5000, 1800)
        population = st.number_input("Population Density (per km²)", 1, 1000, 120)

        river_distance = st.number_input("Distance to River (km)", 0.0, 20.0, 5.0)
        road_distance = st.number_input("Distance to Road (km)", 0.0, 20.0, 3.0)

        ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.45)

        seismic = st.selectbox("Seismic Zone", ["I", "II", "III", "IV", "V"])
        deforestation = st.selectbox("Deforestation Risk", ["Low", "Medium", "High", "Very High"])
        water = st.selectbox("Water Pollution Risk", ["Low", "Medium", "High", "Very High"])
        air = st.selectbox("Air Pollution Risk", ["Low", "Medium", "High", "Very High"])

        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state["input_data"] = pd.DataFrame([{
            "Elevation_m": elevation,
            "Slope_deg": slope,
            "Forest_Cover_Percent": forest_cover,
            "Protected_Area": protected,
            "Wildlife_Sanctuary": wildlife,
            "Annual_Rainfall_mm": rainfall,
            "Population_Density_per_km2": population,
            "Distance_to_River_km": river_distance,
            "Distance_to_Road_km": road_distance,
            "NDVI": ndvi,
            "Seismic_Zone": seismic,
            "Deforestation_Risk": deforestation,
            "Water_Pollution_Risk": water,
            "Air_Pollution_Risk": air
        }])

        st.success("✅ Input saved! Go to **Result** page.")

# PAGE 2: RESULT PAGE
elif page == "Result":
    st.title("📊 Mining Approval Result")

    if st.session_state["input_data"] is None:
        st.warning("⚠️ Please enter inputs first on the Input Details page.")
    else:
        input_df = st.session_state["input_data"]

        st.subheader("🔎 Input Summary")
        st.dataframe(input_df)

        if st.button("Run Mining Approval Model"):
            prediction = model.predict(input_df)[0]

            if prediction == "Yes":
                st.success("✅ MINING APPROVED")
            else:
                st.error("❌ MINING DENIED")

            st.markdown("---")
            st.markdown("### 🧠 Model Decision")
            st.write(f"**Prediction Output:** `{prediction}`")

