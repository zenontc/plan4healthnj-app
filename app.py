import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- 2. Define Drivers and Outcomes (Updated) ---
# Physical Inactivity is now strictly a DRIVER
drivers = [
    'Median HH Income',
    'Crime Index',
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Utilities services threat crude prevalence (%)',
    'Social isolation crude prevalence (%)',
    'Physical inactivity crude prevalence (%)' 
]

# Physical Inactivity has been removed from OUTCOMES
outcomes = [
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequnt physical distress crude prevalence (%)', 
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# --- 3. Helper Functions ---
def get_clean_label(col):
    mapping = {
        'Fair or poor health crude prevalence (%)': 'Poor Health (%)',
        'Frequnt physical distress crude prevalence (%)': 'Physical Distress (%)',
        'Median HH Income': 'Median HH Income ($)',
        'Crime Index': 'Crime Index',
        'Physical inactivity crude prevalence (%)': 'Physical Inactivity (%)'
    }
    if col in mapping:
        return mapping[col]
    return col.replace(" crude prevalence (%)", " (%)")

# --- 4. Regression Engine ---
@st.cache_resource
def get_coefficients(_df, _drivers, _outcomes):
    coeffs = {}
    for outcome in _outcomes:
        coeffs[outcome] = {}
        for driver in _drivers:
            # Ensure we only regress if columns exist and aren't empty
            temp_df = _df[[driver, outcome]].dropna()
            if not temp_df.empty:
                X = temp_df[[driver]]
                y = temp_df[outcome]
                model = LinearRegression().fit(X, y)
                coeffs[outcome][driver] = model.coef_[0]
            else:
                coeffs[outcome][driver] = 0.0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 5. UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health & Planning Tool")
st.title("NJ Municipal Planning and Public Health Tool")
st.markdown("""
This tool models how changes in factors that can be impacted through Planning might impact public health outcomes in New Jersey municipalities.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())

if 'last_muni' not in st.session_state:
    st.session_state.last_muni = muni_list[0]

selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

# Municipality Change Reset
if selected_muni != st.session_state.last_muni:
    for d in drivers:
        if f"slider_{d}" in st.session_state:
            del st.session_state[f"slider_{d}"]
    st.session_state.last_muni = selected_muni

baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Drivers
st.sidebar.header("2. Adjust Factors")
st.sidebar.markdown("Use sliders to show changes.")

if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

adjustment_deltas = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.2), 1000.0
        fmt = "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f"
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    val = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v,
        max_value=max_v,
        key=f"slider_{driver}",
        step=step,
        format=fmt
    )
    adjustment_deltas[driver] = val - current_val

# --- 6. Inter-Driver Logic (Ripple Effects) ---
final_deltas = adjustment_deltas.copy()

# Transportation ripple effects
final_deltas['Food insecurity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.3
final_deltas['Physical inactivity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.3
final_deltas['Social isolation crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude
