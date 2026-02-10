import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

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
st.markdown("""
This tool models how changes in factors that can be impacted through Planning might impact public health outcomes in New Jersey municipalities.
""")

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
st.sidebar.markdown("Use sliders to show changes. Physical Inactivity to Social Isolation factors are the percentage of the population over 18 affected by the issue.")

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

    val = st.sidebar.slider(get_clean_label(driver), min_v, max_v, key=f"slider_{driver}", step=step, format=fmt)
    adjustment_deltas[driver] = val - current_val

# --- 6. Inter-Driver Logic (Conservative Ripple Effects) ---
final_deltas = adjustment_deltas.copy()

# Transportation affects Insecurity and Inactivity (Tightened weights)
final_deltas['Food insecurity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.15
final_deltas['Physical inactivity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.20
final_deltas['Social isolation crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.10

# Income ripple effects (Scaled to be more conservative)
income_delta_scaled = adjustment_deltas['Median HH Income'] / 20000 
final_deltas['Housing insecurity crude prevalence (%)'] -= income_delta_scaled * 0.4
final_deltas['Food insecurity crude prevalence (%)'] -= income_delta_scaled * 0.2

# Crime ripple effect on Physical Inactivity
crime_delta_scaled = adjustment_deltas['Crime Index'] / 20
final_deltas['Physical inactivity crude prevalence (%)'] += crime_delta_scaled * 0.15

# --- 7. Calculate Outcomes with Non-Linear Damping ---
DAMPING_FACTOR = 0.35 
predicted_values = {}

for outcome in outcomes:
    total_linear_change = 0
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        total_linear_change += final_deltas[driver] * coeff
    
    base_val = baseline_data[outcome]
    
    # Non-linear "Resistance" Logic:
    # If the change is negative (improvement), we damp it more as it gets closer to zero.
    if total_linear_change < 0:
        # Use an asymptotic approach so it can't hit zero unless sliders are zeroed
        resistance = (base_val / (base_val + 5)) # Higher base_val = less resistance initially
        actual_change = total_linear_change * DAMPING_FACTOR * resistance
    else:
        actual_change = total_linear_change * DAMPING_FACTOR

    # Floor logic: prevent 0 unless specifically earned through extreme slider movement
    predicted_values[outcome] = max(base_val * 0.1, base_val + actual_change)

# --- 8. Visualizations ---
results = []
for outcome in outcomes:
    results.append({
        "Measure": get_clean_label(outcome).replace(" (%)", ""),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_values[outcome]
    })
    
results_df = pd.DataFrame(results)
fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))

fig.update_layout(
    title=f"Projected Health Outcomes: {selected_muni}",
    yaxis_title="Prevalence (%)",
    barmode='group',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- 9. Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))

for i, outcome in enumerate(outcomes):
    base = baseline_data[outcome]
    pred = predicted_values[outcome]
    diff = pred - base
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None # Gray out very small changes
    
    with cols[i]:
        st.metric(
            label=get_clean_label(outcome),
            value=f"{pred:.1f}%",
            delta=delta_val,
            delta_color="inverse"
        )

st.markdown("---")
st.caption("*Note: This simulator uses a non-linear damping model to reflect the difficulty of eliminating chronic disease prevalence. Driver interdependencies are modeled to show how environmental factors influence behavior.*")
