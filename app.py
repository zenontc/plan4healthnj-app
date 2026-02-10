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
    if col in mapping: return mapping[col]
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
                model = LinearRegression().fit(temp_df[[driver]], temp_df[outcome])
                coeffs[outcome][driver] = model.coef_[0]
            else:
                coeffs[outcome][driver] = 0.0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 5. UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Planning and Public Health Tool")
st.title("NJ Municipal Planning and Public Health Tool")
st.markdown("This tool models how changes in planning factors might impact public health outcomes in NJ municipalities.")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())

if 'last_muni' not in st.session_state:
    st.session_state.last_muni = muni_list[0]

selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

if selected_muni != st.session_state.last_muni:
    for d in drivers:
        if f"slider_{d}" in st.session_state: del st.session_state[f"slider_{d}"]
    st.session_state.last_muni = selected_muni

baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Drivers
st.sidebar.header("2. Adjust Factors")
st.sidebar.markdown("Use sliders to show changes.")

if st.sidebar.button("Reset to Baseline"):
    for d in drivers: st.session_state[f"slider_{d}"] = float(baseline_data[d])

adjustment_deltas = {}
current_slider_values = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    if "Income" in driver:
        min_v, max_v, step, fmt = 0.0, float(df[driver].max() * 1.5), 1000.0, "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step, fmt = 0.0, max(float(df[driver].max()), current_val * 2), 0.1, "%.1f"
    else:
        min_v, max_v, step, fmt = 0.0, max(float(df[driver].max()), current_val * 2), 0.1, "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    val = st.sidebar.slider(get_clean_label(driver), min_v, max_v, key=f"slider_{driver}", step=step, format=fmt)
    current_slider_values[driver] = val
    adjustment_deltas[driver] = val - current_val

# --- 6. Inter-Driver Logic (Ripple Effects) ---
# We use the current values to calculate interdependent shifts
final_deltas = adjustment_deltas.copy()

# Transportation affects Food, Inactivity, and Isolation
final_deltas['Food insecurity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.25
final_deltas['Physical inactivity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.25
final_deltas['Social isolation crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.15

# Income affects Housing and Food
income_delta_scaled = adjustment_deltas['Median HH Income'] / 10000 
final_deltas['Housing insecurity crude prevalence (%)'] -= income_delta_scaled * 0.4
final_deltas['Food insecurity crude prevalence (%)'] -= income_delta_scaled * 0.2

# Crime ripple effect on Physical Inactivity
final_deltas['Physical inactivity crude prevalence (%)'] += (adjustment_deltas['Crime Index'] / 20) * 0.2

# --- 7. Calculate Outcomes with Conservative Damping ---
predicted_values = {}
DAMPING_FACTOR = 0.35 # More conservative

for outcome in outcomes:
    raw_change = 0
    for driver in drivers:
        raw_change += final_deltas[driver] * coefficients[outcome][driver]
    
    # Apply damping to the change
    damped_change = raw_change * DAMPING_FACTOR
    
    # NON-LINEAR FLOOR LOGIC:
    # Instead of Baseline + Change, we use a logistic-style floor.
    # This ensures the outcome approaches zero but never hits it unless drivers are zero.
    baseline = baseline_data[outcome]
    
    if damped_change < 0:
        # If health is improving (change is negative), apply diminishing returns
        # This makes it harder to reach 0% prevalence
        new_val = baseline * np.exp(damped_change / baseline)
    else:
        new_val = baseline + damped_change
        
    predicted_values[outcome] = max(0.1, new_val) # Minimum floor of 0.1%

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
fig.update_layout(title=f"Projected Health Outcomes: {selected_muni}", yaxis_title="Prevalence (%)", barmode='group', height=500)
st.plotly_chart(fig, use_container_width=True)

# --- 9. Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))
for i, outcome in enumerate(outcomes):
    base, pred = baseline_data[outcome], predicted_values[outcome]
    diff = pred - base
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None
    with cols[i]:
        st.metric(label=get_clean_label(outcome), value=f"{pred:.1f}%", delta=delta_val, delta_color="inverse")

st.markdown("---")
st.caption("*Note: This tool uses conservative non-linear modeling. Improvements follow a diminishing returns curve, ensuring health outcomes stay grounded in municipal reality.*")
