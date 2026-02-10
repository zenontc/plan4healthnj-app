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

# Note: Poor Health is handled separately as a composite
core_outcomes = [
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequnt physical distress crude prevalence (%)', 
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)'
]
outcomes = core_outcomes + ['Fair or poor health crude prevalence (%)']

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

# --- 4. Regression Engine (Used for weighting factors) ---
@st.cache_resource
def get_coefficients(_df, _drivers, _outcomes):
    coeffs = {}
    for outcome in _outcomes:
        coeffs[outcome] = {}
        for driver in _drivers:
            temp_df = _df[[driver, outcome]].dropna()
            if not temp_df.empty:
                model = LinearRegression().fit(temp_df[[driver]], temp_df[outcome])
                coeffs[outcome][driver] = abs(model.coef_[0]) # Use magnitude for weighting
            else:
                coeffs[outcome][driver] = 0.1
    return coeffs

coefficients = get_coefficients(df, drivers, core_outcomes)

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
if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

current_values = {}
for driver in drivers:
    base_val = float(baseline_data[driver])
    if "Income" in driver:
        min_v, max_v, step, fmt = 0.0, float(df[driver].max() * 1.5), 1000.0, "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step, fmt = 0.0, max(float(df[driver].max()), base_val * 2), 0.1, "%.1f"
    else:
        min_v, max_v, step, fmt = 0.0, 100.0, 0.1, "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = base_val

    current_values[driver] = st.sidebar.slider(get_clean_label(driver), min_v, max_v, key=f"slider_{driver}", step=step, format=fmt)

# --- 6. Inter-Driver Logic (Ripple Effects) ---
# We calculate a "Effective Delta" for each driver
effective_values = current_values.copy()
# Transp impacts Activity and Food
effective_values['Physical inactivity crude prevalence (%)'] += (current_values['Transportation barriers crude prevalence (%)'] - baseline_data['Transportation barriers crude prevalence (%)']) * 0.2
effective_values['Food insecurity crude prevalence (%)'] += (current_values['Transportation barriers crude prevalence (%)'] - baseline_data['Transportation barriers crude prevalence (%)']) * 0.2

# --- 7. Conservative Proportional Calculation ---
predicted_values = {}

for outcome in core_outcomes:
    # 1. Calculate weighted influence of each driver
    total_impact_score = 0
    baseline_impact_score = 0
    
    for driver in drivers:
        if "Income" in driver: continue # Income acts as a secondary scaler
        
        weight = coefficients[outcome][driver]
        # Boost specific relationships
        if outcome == 'Obesity crude prevalence (%)' and driver == 'Physical inactivity crude prevalence (%)': weight *= 3.0
        if outcome == 'Diabetes crude prevalence (%)' and driver == 'Physical inactivity crude prevalence (%)': weight *= 2.5
        if outcome == 'Depression crude prevalence (%)' and driver == 'Social isolation crude prevalence (%)': weight *= 3.0
        
        total_impact_score += max(0, effective_values[driver]) * weight
        baseline_impact_score += max(0.1, baseline_data[driver]) * weight
    
    # 2. Multiplier represents the aggregate improvement/decline
    multiplier = total_impact_score / baseline_impact_score
    
    # 3. Apply Income scaling (Higher income dampens the multiplier slightly)
    income_ratio = baseline_data['Median HH Income'] / max(1, current_values['Median HH Income'])
    multiplier = multiplier * (0.8 + 0.2 * income_ratio)
    
    # 4. Conservative Damping
    final_multiplier = 1.0 + ((multiplier - 1.0) * 0.4)
    predicted_values[outcome] = baseline_data[outcome] * final_multiplier

# --- 8. Poor Health Composite Logic ---
# Poor health is the weighted average of core outcomes to ensure it doesn't zero out prematurely
avg_outcome_change = sum(predicted_values[o] for o in core_outcomes) / sum(baseline_data[o] for o in core_outcomes)
predicted_values['Fair or poor health crude prevalence (%)'] = baseline_data['Fair or poor health crude prevalence (%)'] * avg_outcome_change

# --- 9. Visualizations & Metrics ---
results_df = pd.DataFrame([{
    "Measure": get_clean_label(o).replace(" (%)", ""),
    "Baseline": baseline_data[o],
    "Simulated": predicted_values[o]
} for o in outcomes])

fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))
fig.update_layout(title=f"Projected Health Outcomes: {selected_muni}", yaxis_title="Prevalence (%)", barmode='group', height=500)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))
for i, outcome in enumerate(outcomes):
    base, pred = baseline_data[outcome], predicted_values[outcome]
    diff = pred - base
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None
    with cols[i]:
        st.metric(label=get_clean_label(outcome), value=f"{pred:.1f}%", delta=delta_val, delta_color="inverse")

st.markdown("---")
st.caption("*Note: This model uses proportional scaling and interdependency logic. Outcomes are tied to the aggregate state of all planning factors to ensure conservative and realistic simulations.*")
