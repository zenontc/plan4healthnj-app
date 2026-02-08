import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    # Load the pre-calculated municipal data
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'NJ_Municipal_Health_Data.csv' not found. Please ensure it is in the same directory.")
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

# Mapping for cleaner labels using "Crude Prevalence"
labels = {
    'Median HH Income': 'Median Household Income ($)',
    'Crime Index': 'Crime Index (AGS)',
    'Transportation barriers crude prevalence (%)': 'Transportation Barriers (Crude Prevalence)',
    'Food insecurity crude prevalence (%)': 'Food Insecurity (Crude Prevalence)',
    'Housing insecurity crude prevalence (%)': 'Housing Insecurity (Crude Prevalence)',
    'Utilities services threat crude prevalence (%)': 'Utility Service Threats (Crude Prevalence)',
    'Social isolation crude prevalence (%)': 'Social Isolation (Crude Prevalence)',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress (Crude Prevalence)',
    'Depression crude prevalence (%)': 'Depression (Crude Prevalence)'
}

def get_label(col):
    # Standardize to "Crude Prevalence" for display
    return labels.get(col, col.replace(" crude prevalence (%)", " (Crude Prevalence)"))

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
**Instructions:** Select a municipality. The sliders are set to the **Actual Baseline Values** for that location. 
* Slide **Left** to simulate a decrease in that factor.
* Slide **Right** to simulate an increase.
