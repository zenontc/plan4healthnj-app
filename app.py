import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    # Ensure NJ_Municipal_Health_Data.csv is in the same folder
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'NJ_Municipal_Health_Data.csv' not found.")
    st.stop()

# --- 2. Define Drivers and Outcomes ---
drivers = [
    'Median HH Income',
    'Crime Index',
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Utilities services threat crude prevalence (%)',
    'Social isolation crude prevalence (%)'
]

outcomes = [
    'Physical inactivity crude prevalence (%)',
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequnt physical distress crude prevalence (%)', 
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# Mapping for cleaner labels (using "Rate" instead of "%")
labels = {
    'Median HH Income': 'Median Household Income ($)',
    'Crime Index': 'Crime Index (AGS)',
    'Transportation barriers crude prevalence (%)': 'Transportation Barriers (Rate per 1,000)',
    'Food insecurity crude prevalence (%)': 'Food Insecurity (Rate per 1,000)',
    'Housing insecurity crude prevalence (%)': 'Housing Insecurity (Rate per 1,000)',
    'Utilities services threat crude prevalence (%)': 'Utility Service Threats (Rate per 1,000)',
    'Social isolation crude prevalence (%)': 'Social Isolation (Rate per 1,000)',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress (Rate per 1,000)',
    'Depression crude prevalence (%)': 'Depression (Rate per 1,000)'
}

def get_label(col):
    return labels.get(col, col.replace(" crude prevalence (%)", " (Rate per 1,000)"))

# --- 3. Statistical Relationships ---
coefficients = {}
for outcome in outcomes:
    coefficients[outcome] = {}
    for driver in drivers:
        X = df[[driver]].fillna(df[driver].mean())
        y = df[outcome].fillna(df[outcome].mean())
        model = LinearRegression().fit(X, y)
        coefficients[outcome][driver] = model.coef_[0]

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health Simulator")

st.title("NJ Municipal Health & Planning Simulator")
st.markdown("""
**Instructions:** Select a municipality. The sliders below are set to the **actual current values** for that area. 
Adjust them to see how changes in social drivers impact health outcomes.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

# Get baseline data for selected muni
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

st.sidebar.header("2. Planning Factors (Actuals)")

# --- SLIDER LOGIC (REPLACED 100% WITH ACTUALS) ---
adjustment_deltas = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    
    #
