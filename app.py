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

df = load_data()

# --- 2. Define Drivers (Inputs) and Outcomes (Outputs) ---
drivers = [
    'Crime Index',
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Social isolation crude prevalence (%)'
]

outcomes = [
    'Physical inactivity crude prevalence (%)',
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# --- 3. Calculate Statistical Relationships (Simple Linear Regression) ---
# We calculate the sensitivity of each outcome to each driver individually.
# This coefficient represents "How much does Outcome Y change if Driver X increases by 1 unit?"
coefficients = {}

for outcome in outcomes:
    coefficients[outcome] = {}
    for driver in drivers:
        # Fit a simple linear regression for each pair
        X = df[[driver]].fillna(df[driver].mean())
        y = df[outcome].fillna(df[outcome].mean())
        model = LinearRegression().fit(X, y)
        coefficients[outcome][driver] = model.coef_[0]

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

# Get baseline data for selected muni
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Adjust Drivers
st.sidebar.header("2. Adjust Planning Factors")
st.sidebar.markdown("Use sliders to simulate improvements or decline.")

adjustment_deltas = {}

for driver in drivers:
    current_val = baseline_data[driver]
    
    # Create a slider centered at the current value
    min_val = 0.0
    max_val = max(float(df[driver].max()), current_val * 2)
    
    new_val = st.sidebar.slider(
        f"{driver}",
        min_value=min_val,
        max_value=max_val,
        value=float(current_val),
        format="%.1f"
    )
    
    # Calculate the change (delta)
    adjustment_deltas[driver] = new_val - current_val

# --- 5. Calculate Predicted Outcomes ---
predicted_values = {}

for outcome in outcomes:
    total_change = 0
    # We sum the effects of all driver changes
    # Note: This assumes effects are additive, which is a simplification for simulation purposes.
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        delta = adjustment_deltas[driver]
        total_change += delta * coeff
    
    # Apply change to baseline
    baseline_val = baseline_data[outcome]
    predicted_val = max(0, baseline_val + total_change) # Ensure no negative prevalence
    predicted_values[outcome] = predicted_val

# --- 6. Visualization ---

# Comparison Data Structure
results = []
for outcome in outcomes:
    results.append({
        "Measure": outcome.replace(" crude prevalence (%)", ""),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_values[outcome]
    })
    
results_df = pd.DataFrame(results)

# Plotly Bar Chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=results_df["Measure"],
    y=results_df["Baseline"],
    name='Current Baseline',
    marker_color='lightslategray'
))

fig.add_trace(go.Bar(
    x=results_df["Measure"],
    y=results_df["Simulated"],
    name='Simulated Scenario',
    marker_color='royalblue'
))

fig.update_layout(
    title=f"Projected Health Outcomes for {selected_muni}",
    yaxis_title="Prevalence (%)",
    barmode='group',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# --- 7. Detailed Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))

for i, outcome in enumerate(outcomes):
    baseline = baseline_data[outcome]
    predicted = predicted_values[outcome]
    change = predicted - baseline
    
    with cols[i]:
        st.metric(
            label=outcome.replace(" crude prevalence (%)", ""),
            value=f"{predicted:.1f}%",
            delta=f"{change:.1f}%",
            delta_color="inverse" # Negative change (drop in disease) is green (good)
        )

st.markdown("---")
st.caption("*Note: This model uses simple linear regression coefficients derived from NJ municipal data. It assumes additive effects and is for planning simulation purposes only.*")