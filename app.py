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

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- 2. Configuration: Drivers & Outcomes ---
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
    'Frequnt physical distress crude prevalence (%)', # Typo in CSV
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# --- 3. Advanced Modeling (Multiple Regression for Conservatism) ---
@st.cache_resource
def train_models(_df, drivers, outcomes):
    models = {}
    # Train a Multiple Regression for each outcome to be more conservative
    for outcome in outcomes:
        valid_df = _df[drivers + [outcome]].dropna()
        X = valid_df[drivers]
        y = valid_df[outcome]
        models[outcome] = LinearRegression().fit(X, y)
    
    # Cascade Model: Transportation impacts Food Insecurity
    # We'll calculate how much Food Insecurity changes per unit of Transportation
    cascade_df = _df[['Transportation barriers crude prevalence (%)', 'Food insecurity crude prevalence (%)']].dropna()
    cascade_model = LinearRegression().fit(cascade_df[['Transportation barriers crude prevalence (%)']], cascade_df['Food insecurity crude prevalence (%)'])
    
    return models, cascade_model

models, cascade_model = train_models(df, drivers, outcomes)

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health Simulator")

# Session State for Resetting Sliders
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_sliders():
    st.session_state.reset_key += 1

st.title("NJ Municipal Health & Planning Simulator")
st.sidebar.button("Reset Sliders to Baseline", on_click=reset_sliders)

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Sliders
st.sidebar.header("2. Adjust Planning Factors")
adj_vals = {}

# We create sliders in the specific order requested
slider_order = [
    ('Median HH Income', "$", 500.0),
    ('Crime Index', "", 1.0),
    ('Transportation barriers crude prevalence (%)', "%", 0.1),
    ('Food insecurity crude prevalence (%)', "%", 0.1),
    ('Housing insecurity crude prevalence (%)', "%", 0.1),
    ('Utilities services threat crude prevalence (%)', "%", 0.1),
    ('Social isolation crude prevalence (%)', "%", 0.1),
]

for driver, unit, step in slider_order:
    base_val = float(baseline_data[driver])
    max_range = max(float(df[driver].max()), base_val * 2)
    
    # Display logic for "Income" vs others
    label = driver.replace(" crude prevalence (%)", "")
    
    adj_vals[driver] = st.sidebar.slider(
        label,
        min_value=0.0,
        max_value=max_range,
        value=base_val,
        step=step,
        format=f"{unit}%.1f" if unit == "$" else "%.1f",
        key=f"{driver}_{st.session_state.reset_key}"
    )

# --- 5. Calculation Logic with Cascading Effect ---
# 1. Calculate Transportation Delta
trans_delta = adj_vals['Transportation barriers crude prevalence (%)'] - baseline_data['Transportation barriers crude prevalence (%)']

# 2. Update Food Insecurity based on Transportation (Cascade)
# If user changed Transportation, it slightly pushes Food Insecurity
food_insecure_cascade = trans_delta * cascade_model.coef_[0]
effective_inputs = pd.DataFrame([adj_vals])
# We add the cascade effect to the food insecurity input for the final calculation
effective_inputs['Food insecurity crude prevalence (%)'] += food_insecure_cascade

# 3. Predict Health Outcomes
predicted_results = {}
for outcome in outcomes:
    pred = models[outcome].predict(effective_inputs)[0]
    predicted_results[outcome] = max(0, pred)

# --- 6. Visualization ---
results_mapping = {
    'Physical inactivity crude prevalence (%)': 'Physical Inactivity',
    'Obesity crude prevalence (%)': 'Obesity',
    'Diabetes crude prevalence (%)': 'Diabetes',
    'Frequnt physical distress crude prevalence (%)': 'Phys. Distress',
    'Depression crude prevalence (%)': 'Depression',
    'Frequent mental distress crude prevalence (%)': 'Mental Distress',
    'Fair or poor health crude prevalence (%)': 'Poor Health'
}

chart_data = []
for outcome in outcomes:
    chart_data.append({
        "Measure": results_mapping[outcome],
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_results[outcome]
    })
results_df = pd.DataFrame(chart_data)

fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))
fig.update_layout(title=f"Projected Health Outcomes: {selected_muni}", barmode='group', height=450)
st.plotly_chart(fig, use_container_width=True)

# --- 7. Detailed Metrics ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))

for i, outcome in enumerate(outcomes):
    base = baseline_data[outcome]
    sim = predicted_results[outcome]
    diff = sim - base
    
    with cols[i]:
        # Logic for gray 0.0% with no arrow
        if abs(diff) < 0.01:
            st.metric(label=results_mapping[outcome], value=f"{sim:.1f}%", delta=None)
        else:
            st.metric(
                label=results_mapping[outcome], 
                value=f"{sim:.1f}%", 
                delta=f"{diff:.1f}%", 
                delta_color="inverse"
            )

st.caption("Note: This model uses Multiple Linear Regression and a cascading link between Transportation and Food Insecurity.")
