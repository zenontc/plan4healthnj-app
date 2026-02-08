import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    # Load the pre-calculated municipal data
    # Ensure NJ_Municipal_Health_Data.csv is in the same folder
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'NJ_Municipal_Health_Data.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# --- 2. Define Drivers (Inputs) and Outcomes (Outputs) ---
# Updated list with Income and Utilities
drivers = [
    'Median HH Income',
    'Crime Index',
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Utilities services threat crude prevalence (%)',
    'Social isolation crude prevalence (%)'
]

# Updated list with Physical Distress and Depression
outcomes = [
    'Physical inactivity crude prevalence (%)',
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequnt physical distress crude prevalence (%)', # Note: "Frequnt" matches the CSV typo
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# Mapping for cleaner labels in the UI
labels = {
    'Median HH Income': 'Median Household Income',
    'Crime Index': 'Crime Index (AGS)',
    'Transportation barriers crude prevalence (%)': 'Transportation Barriers',
    'Food insecurity crude prevalence (%)': 'Food Insecurity',
    'Housing insecurity crude prevalence (%)': 'Housing Insecurity',
    'Utilities services threat crude prevalence (%)': 'Utility Service Threats',
    'Social isolation crude prevalence (%)': 'Social Isolation',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress', # Fix typo for display
    'Depression crude prevalence (%)': 'Depression'
}

def get_label(col):
    return labels.get(col, col.replace(" crude prevalence (%)", ""))

# --- 3. Calculate Statistical Relationships (Simple Linear Regression) ---
# We calculate the coefficient: How much does Outcome change per 1 unit of Driver?
coefficients = {}

for outcome in outcomes:
    coefficients[outcome] = {}
    for driver in drivers:
        # Fit a simple linear regression for each pair
        # Fill NA with mean to ensure stability
        X = df[[driver]].fillna(df[driver].mean())
        y = df[outcome].fillna(df[outcome].mean())
        model = LinearRegression().fit(X, y)
        coefficients[outcome][driver] = model.coef_[0]

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health & Planning Tool")

st.title("NJ Municipal Health & Planning Simulator")
st.markdown("""
**Instructions:** Select a municipality and adjust the "Planning Factors" sliders. 
The sliders are set to **100%** (Current Baseline). 
* Move **Left (<100%)** to simulate a reduction (e.g., less crime, fewer barriers).
* Move **Right (>100%)** to simulate an increase.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

# Get baseline data for selected muni
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Adjust Drivers
st.sidebar.header("2. Planning Factors")
st.sidebar.markdown("Adjust percentage relative to baseline (100%).")

# --- RESET BUTTON LOGIC ---
# Initialize session state for sliders if not present
if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = False

def reset_sliders():
    for driver in drivers:
        st.session_state[f"slider_{driver}"] = 100.0

st.sidebar.button("Reset to Baseline", on_click=reset_sliders)

# Container for Simulation Inputs
adjustment_deltas = {}

for driver in drivers:
    # Key for session state
    key = f"slider_{driver}"
    
    # Initialize key if not exists
    if key not in st.session_state:
        st.session_state[key] = 100.0

    # Get current baseline value to display
    current_val_abs = baseline_data[driver]
    
    # Format label differently for Income ($) vs Prevalence (%)
    if "Income" in driver:
        val_fmt = f"${current_val_abs:,.0f}"
    else:
        val_fmt = f"{current_val_abs:.1f}%"

    st.sidebar.markdown(f"**{get_label(driver)}** (Base: {val_fmt})")
    
    # The Slider (0% to 200%)
    pct_val = st.sidebar.slider(
        f"Adjust {get_label(driver)}",
        min_value=0.0,
        max_value=200.0,
        step=5.0,
        key=key,
        label_visibility="collapsed"
    )
    
    # Calculate the simulated absolute value
    # Formula: Simulated = Baseline * (Percentage / 100)
    simulated_val_abs = current_val_abs * (pct_val / 100.0)
    
    # Calculate the Delta (Simulated - Baseline)
    # This delta is what we multiply by the regression coefficient
    adjustment_deltas[driver] = simulated_val_abs - current_val_abs


# --- 5. Calculate Predicted Outcomes ---
predicted_values = {}

for outcome in outcomes:
    total_change = 0
    # Sum effects of all driver changes
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        delta = adjustment_deltas[driver]
        total_change += delta * coeff
    
    # Apply change to baseline
    baseline_val = baseline_data[outcome]
    predicted_val = max(0, baseline_val + total_change) # Prevent negative values
    predicted_values[outcome] = predicted_val

# --- 6. Visualization ---

# Create comparison dataframe
results = []
for outcome in outcomes:
    results.append({
        "Measure": get_label(outcome),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_values[outcome]
    })
    
results_df = pd.DataFrame(results)

# Plotly Grouped Bar Chart
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
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- 7. Detailed Metrics Table ---
st.subheader("Detailed Projections")

# Dynamic columns based on number of outcomes
cols = st.columns(len(outcomes))

for i, outcome in enumerate(outcomes):
    baseline = baseline_data[outcome]
    predicted = predicted_values[outcome]
    change = predicted - baseline
    
    # Color logic: Negative change (healthier) = Green ("inverse" in st.metric)
    # Positive change (less healthy) = Red ("normal")
    # For Income, this would be reversed, but here outcomes are all "Prevalence of X", so dropping is good.
    
    with cols[i]:
        st.metric(
            label=get_label(outcome),
            value=f"{predicted:.1f}%",
            delta=f"{change:.1f}%",
            delta_color="inverse" 
        )

st.markdown("---")
st.caption("*Note: 'Frequnt Physical Distress' retains the spelling from the source dataset.*")

