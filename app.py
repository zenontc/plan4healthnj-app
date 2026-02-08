import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('NJ_Municipal_Health_Data.csv')
        # Clean column names just in case there are leading/trailing spaces
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("🚨 **File Not Found:** Please make sure 'NJ_Municipal_Health_Data.csv' is in the same folder as this script.")
    st.stop()

# --- 2. Define Drivers and Outcomes ---
# I have kept the names exactly as you provided.
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

# --- 3. DIAGNOSTIC CHECK ---
# This ensures the app doesn't crash if names don't match exactly
missing_cols = [c for c in drivers + outcomes if c not in df.columns]
if missing_cols:
    st.error("🚨 **Column Name Mismatch!**")
    st.write("The following names in the code do not exist in your CSV:")
    st.write(missing_cols)
    st.write("Current CSV Columns:", list(df.columns))
    st.stop()

# --- 4. Mapping for UI Labels ---
labels = {
    'Median HH Income': 'Median HH Income ($)',
    'Crime Index': 'Crime Index',
    'Transportation barriers crude prevalence (%)': 'Transportation Barriers',
    'Food insecurity crude prevalence (%)': 'Food Insecurity',
    'Housing insecurity crude prevalence (%)': 'Housing Insecurity',
    'Utilities services threat crude prevalence (%)': 'Utility Service Threats',
    'Social isolation crude prevalence (%)': 'Social Isolation',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress',
    'Depression crude prevalence (%)': 'Depression'
}

def get_label(col):
    return labels.get(col, col.replace(" crude prevalence (%)", ""))

# --- 5. Calculate Statistical Relationships ---
@st.cache_resource
def get_coefficients(_df, drivers, outcomes):
    coeffs = {}
    for outcome in outcomes:
        coeffs[outcome] = {}
        for driver in drivers:
            # Drop NaNs for the regression calculation
            temp_df = _df[[driver, outcome]].dropna()
            if len(temp_df) > 1:
                X = temp_df[[driver]]
                y = temp_df[outcome]
                model = LinearRegression().fit(X
