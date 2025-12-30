# -*- coding: utf-8 -*-
"""
Finalized Mining Assessment App - 2025
Biodiversity, Infrastructure & Impact Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="AI Mining Assessment 2025",
    page_icon="⛰️",
    layout="wide"
)

st.title("⛰️ AI-Driven Mining Assessment System (2025)")
st.markdown(
    "Evaluates **legal, environmental, safety & AI-policy constraints** "
    "to approve or reject mining operations."
)

# ------------------------------------
# DATA UPLOAD
# ------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Mining Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload the mining CSV file to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file, low_memory=False)
st.success("Dataset loaded successfully")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ------------------------------------
# 1. PREPROCESSING
# ------------------------------------
risk_mapping = {'Low': 2, 'Medium': 5, 'High': 8, 'Very High': 10}
for col in ['Deforestation_Risk', 'Water_Pollution_Risk', 'Air_Pollution_Risk']:
    df[col] = df[col].fillna('Low').map(risk_mapping)

df['Environmental_Risk_Index'] = (
    df[['Deforestation_Risk', 'Water_Pollution_Risk', 'Air_Pollution_Risk']]
    .mean(axis=1) / 10.0
)

seismic_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'Unknown': 0}
df['Seismic_Zone'] = df['Seismic_Zone'].astype(str).map(seismic_mapping).fillna(0)

le = LabelEncoder()
df['Mining_Allowed_Numeric'] = le.fit_transform(
    df['Mining_Allowed'].astype(str)
)

# ------------------------------------
# 2. AI MODEL TRAINING
# ------------------------------------
features = [
    'Elevation_m', 'Slope_deg', 'Forest_Cover_Percent',
    'Protected_Area', 'Annual_Rainfall_mm', 'Seismic_Zone',
    'Population_Density_per_km2', 'Distance_to_River_km',
    'Distance_to_Road_km', 'NDVI'
]

X = df[features].fillna(df[features].median(numeric_only=True))
y = df['Mining_Allowed_Numeric']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

df['AI_Predicted_Allowed'] = model.predict(X_scaled)

st.success("AI model trained and predictions generated")

# ------------------------------------
# 3. DECISION & IMPACT LOGIC (UNCHANGED)
# ------------------------------------
def to_binary(val):
    if pd.isna(val):
        return 0
    return 1 if str(val).strip().lower() in ['1', 'true', 'yes', 'y'] else 0

def evaluate_and_impact(row):
    reasons, impacts = [], []

    is_protected = to_binary(row.get('Protected_Area', 0))
    is_sanctuary = to_binary(row.get('Wildlife_Sanctuary', 0))
    risk_index = row.get('Environmental_Risk_Index', 0)
    slope = row.get('Slope_deg', 0)
    seismic = row.get('Seismic_Zone', 0)
    ai_allowed = row.get('AI_Predicted_Allowed', 1)

    trees = str(row.get('Dominant_Trees', 'Unknown'))
    animals = str(row.get('Key_Animals', 'Unknown'))
    river = str(row.get('Nearby_River', 'None'))
    land_use = str(row.get('Land_Use_Type', 'Unknown'))
    ndvi = row.get('NDVI', 0)

    legal_flag = (is_protected == 1 or is_sanctuary == 1)
    risk_flag = (risk_index > 0.65)
    safety_flag = (slope > 35 and seismic >= 4)
    policy_flag = (ai_allowed == 0)

    if legal_flag:
        status = "DENIED (Legal)"
        reasons.append("Inviolate Zone: Protected Area/Sanctuary present.")
        impacts.append(f"Mining would destroy {trees} canopy and displace {animals}.")
    elif risk_flag:
        status = "REJECTED (High Risk)"
        reasons.append(f"High Risk Index ({risk_index:.2f} > 0.65).")
        impacts.append(f"Mining could contaminate {river} and degrade {land_use} soils.")
    elif safety_flag:
        status = "REJECTED (Safety)"
        reasons.append(f"Unsafe slope {slope}° in Seismic Zone {seismic}.")
        impacts.append("High probability of landslides affecting settlements.")
    elif policy_flag:
        status = "DENIED (Policy)"
        reasons.append("AI Policy Restriction triggered.")
        impacts.append("Conflicts with sustainability masterplans.")
    else:
        status = "APPROVED"
        reasons.append(f"Feasible: NDVI {ndvi:.2f}, road access available.")
        impacts.append("Mitigation protocols required for minimal impact.")

    return pd.Series([status, "; ".join(reasons), "; ".join(impacts)])

df[['Final_Mining_Assessment',
    'Justification',
    'Potential_Environmental_Impacts']] = df.apply(
        evaluate_and_impact, axis=1
    )

# ------------------------------------
# 4. RESULTS DISPLAY
# ------------------------------------
st.subheader("📊 Final Mining Assessment Results")
st.dataframe(
    df[['Final_Mining_Assessment', 'Justification',
        'Potential_Environmental_Impacts']]
)

# Decision distribution
st.subheader("📈 Decision Summary")
fig, ax = plt.subplots(figsize=(8, 4))
df['Final_Mining_Assessment'].value_counts().plot(
    kind='bar', ax=ax
)
ax.set_xlabel("Decision")
ax.set_ylabel("Count")
st.pyplot(fig)

# ------------------------------------
# 5. DOWNLOAD REPORT
# ------------------------------------
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Full Mining Impact Report",
    data=csv_data,
    file_name="Mining_Full_Impact_Report_2025.csv",
    mime="text/csv"
)
