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

slider_values = {}
for driver in drivers:
    current_val = float(baseline_data[driver])
    
    if "Income" in driver:
        # For income, we treat "0" as the worst case for the math
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.5), 1000.0
        fmt = "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f"
    else:
        min_v, max_v, step = 0.0, 100.0, 0.1
        fmt = "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    slider_values[driver] = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v,
        max_value=max_v,
        key=f"slider_{driver}",
        step=step,
        format=fmt
    )

# --- 6. Inter-Driver Logic (Conservative Ripple Effects) ---
# We calculate a "Final Effective Driver Value"
effective_drivers = slider_values.copy()

# Transportation impacts Activity and Food
effective_drivers['Physical inactivity crude prevalence (%)'] += (slider_values['Transportation barriers crude prevalence (%)'] - baseline_data['Transportation barriers crude prevalence (%)']) * 0.15
effective_drivers['Food insecurity crude prevalence (%)'] += (slider_values['Transportation barriers crude prevalence (%)'] - baseline_data['Transportation barriers crude prevalence (%)']) * 0.20

# Crime impacts Activity
effective_drivers['Physical inactivity crude prevalence (%)'] += (slider_values['Crime Index'] - baseline_data['Crime Index']) * 0.05

# Income impacts Housing and Food (Inversely)
income_impact = (slider_values['Median HH Income'] - baseline_data['Median HH Income']) / 10000
effective_drivers['Housing insecurity crude prevalence (%)'] -= income_impact * 0.4
effective_drivers['Food insecurity crude prevalence (%)'] -= income_impact * 0.2

# Ensure no percentage goes below 0 before calculating outcomes
for d in drivers:
    if d != 'Median HH Income' and d != 'Crime Index':
        effective_drivers[d] = max(0, min(100, effective_drivers[d]))

# --- 7. Calculate Outcomes (Conservative Proportional Model) ---
predicted_values = {}

# Calculate a "Global Factor Score" (0 = perfect environment, 1 = baseline)
# This ensures that outcomes only hit zero if the environment is "perfect"
total_baseline_env = sum([baseline_data[d] for d in drivers if d != 'Median HH Income'])
total_current_env = sum([effective_drivers[d] for d in drivers if d != 'Median HH Income'])

# Special handling for Income (High income = better environment)
income_ratio = baseline_data['Median HH Income'] / max(1, effective_drivers['Median HH Income'])
env_ratio = (total_current_env + (income_ratio * 10)) / (total_baseline_env + 10)

for outcome in outcomes:
    # Use Linear Regression to find the direction of change
    total_linear_delta = 0
    for driver in drivers:
        delta = effective_drivers[driver] - baseline_data[driver]
        total_linear_delta += delta * coefficients[outcome][driver]
    
    # Apply Damping (Conservative mix of baseline and linear prediction)
    # 0.4 damping means we take 40% of the statistical change
    damped_change = total_linear_delta * 0.4
    
    # Proportional Scaling: The outcome scales with the environment ratio
    # This prevents the outcome from hitting 0 unless the drivers are 0
    simulated_val = (baseline_data[outcome] + damped_change) * env_ratio
    
    # Absolute Floor: If all drivers are 0, outcome must be 0
    if total_current_env == 0 and effective_drivers['Median HH Income'] > baseline_data['Median HH Income']:
        predicted_values[outcome] = 0
    else:
        predicted_values[outcome] = max(0.1, simulated_val) if total_current_env > 0 else 0

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
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None
    
    with cols[i]:
        st.metric(
            label=get_clean_label(outcome),
            value=f"{pred:.1f}%",
            delta=delta_val,
            delta_color="inverse"
        )

st.markdown("---")
st.caption("*Note: This model uses a proportional decay logic. Health outcomes are anchored to the baseline and only reach zero if all social and environmental drivers are eliminated.*")
