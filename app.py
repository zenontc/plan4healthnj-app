import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- 2. Define Drivers and Outcomes ---
drivers = [
    'Median HH Income',
    'Crime Index',
    'Physical inactivity crude prevalence (%)', 
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Utilities services threat crude prevalence (%)',
    'Social isolation crude prevalence (%)'
]

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
st.set_page_config(layout="wide", page_title="NJ Planning and Public Health Tool")
st.title("NJ Municipal Planning and Public Health Tool")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())

if 'last_muni' not in st.session_state:
    st.session_state.last_muni = muni_list[0]

selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

if selected_muni != st.session_state.last_muni:
    for d in drivers:
        if f"slider_{d}" in st.session_state:
            del st.session_state[f"slider_{d}"]
    st.session_state.last_muni = selected_muni

baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Drivers
st.sidebar.header("2. Adjust Factors")
st.sidebar.markdown("Use sliders to show changes. Physical Inactivity to Social Isolation factors are the percentage of the population over 18 affected.")

if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

slider_values = {}
for driver in drivers:
    current_val = float(baseline_data[driver])
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.5), 1000.0
        fmt = "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f"
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    slider_values[driver] = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v, max_value=max_v,
        key=f"slider_{driver}", step=step, format=fmt
    )

# --- 6. Inter-Driver Logic (The "Ripple" Effect) ---
# We calculate how much the user has moved the slider relative to baseline
deltas = {d: slider_values[d] - baseline_data[d] for d in drivers}

# Ripple: Transport affects Food, Activity, and Social
deltas['Food insecurity crude prevalence (%)'] += deltas['Transportation barriers crude prevalence (%)'] * 0.2
deltas['Physical inactivity crude prevalence (%)'] += deltas['Transportation barriers crude prevalence (%)'] * 0.2
deltas['Social isolation crude prevalence (%)'] += deltas['Transportation barriers crude prevalence (%)'] * 0.1

# Ripple: Income affects Housing/Food
income_impact = (slider_values['Median HH Income'] - baseline_data['Median HH Income']) / 10000
deltas['Housing insecurity crude prevalence (%)'] -= income_impact * 0.4
deltas['Food insecurity crude prevalence (%)'] -= income_impact * 0.2

# --- 7. Conservative Calculation & Floor Logic ---
predicted_values = {}
DAMPING = 0.4

# Check if all non-income factors are zeroed out
social_factors = [d for d in drivers if "Income" not in d]
all_social_zero = all(slider_values[d] <= 0.05 for d in social_factors)

for outcome in outcomes:
    total_impact = 0
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        total_impact += deltas[driver] * coeff
    
    # Apply damping to the change
    change = total_impact * DAMPING
    new_val = baseline_data[outcome] + change
    
    # Logic: Outcomes only go to 0 if social factors are 0 AND Income is high
    if all_social_zero:
        # If social factors are zero, the outcome approaches zero 
        # based on how much Income has increased.
        income_ratio = slider_values['Median HH Income'] / max(df['Median HH Income'])
        new_val = max(0, new_val * (1 - income_ratio * 0.5)) if new_val > 0 else 0
    
    predicted_values[outcome] = max(0, new_val)

# --- 8. Visualizations ---
results_df = pd.DataFrame([{
    "Measure": get_clean_label(o).replace(" (%)", ""),
    "Baseline": baseline_data[o],
    "Simulated": predicted_values[o]
} for o in outcomes])

fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))
fig.update_layout(title=f"Projected Health Outcomes: {selected_muni}", barmode='group', height=500)
st.plotly_chart(fig, use_container_width=True)

# --- 9. Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))
for i, outcome in enumerate(outcomes):
    diff = predicted_values[outcome] - baseline_data[outcome]
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.01 else None
    with cols[i]:
        st.metric(label=get_clean_label(outcome), value=f"{predicted_values[outcome]:.1f}%", 
                  delta=delta_val, delta_color="inverse")

st.markdown("---")
st.caption("*Note: Outcomes are linked through a ratio-based logic. Health improvements are capped by socioeconomic baselines unless all social drivers are addressed.*")
