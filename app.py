import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- 2. Define Drivers (Inputs) and Outcomes (Outputs) ---
# Ordered as requested
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
    'Fair or poor health crude prevalence (%)' # Will rename to "Poor Health" in UI
]

# --- 3. Calculate Statistical Relationships ---
# We use a damping factor (0.4) to make the model more conservative
SENSITIVITY_DAMPING = 0.4 

@st.cache_resource
def get_coefficients(_df, _drivers, _outcomes):
    coeffs = {}
    for outcome in _outcomes:
        coeffs[outcome] = {}
        for driver in _drivers:
            temp_df = _df[[driver, outcome]].dropna()
            if not temp_df.empty:
                X = temp_df[[driver]]
                y = temp_df[outcome]
                model = LinearRegression().fit(X, y)
                # Apply damping to keep results conservative
                coeffs[outcome][driver] = model.coef_[0] * SENSITIVITY_DAMPING
            else:
                coeffs[outcome][driver] = 0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health & Planning Tool")

st.title("NJ Municipal Health & Planning Simulator")
st.markdown("""
This tool simulates how changes in the **built environment** and **social factors** (drivers) 
might impact **public health outcomes** in New Jersey municipalities.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# --- RESET LOGIC ---
if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

def reset_sliders():
    st.session_state.reset_key += 1

st.sidebar.button("Reset Sliders to Baseline", on_click=reset_sliders)

st.sidebar.header("2. Adjust Planning Factors")
st.sidebar.markdown("Use sliders to simulate improvements or decline.")

slider_vals = {}
for driver in drivers:
    current_val = float(baseline_data[driver])
    
    # Range logic
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max()), 1000.0
        fmt = "$
