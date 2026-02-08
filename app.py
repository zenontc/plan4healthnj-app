import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    # Ensure NJ_Municipal_Health_Data.csv is in the same folder
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'NJ_Municipal_Health_Data.csv' not found.")
    st.stop()

# --- 2. Define Drivers and Outcomes ---
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
    'Fair or poor health crude prevalence (%)'
]

# Mapping for cleaner labels (using "Rate" instead of "%")
labels = {
    'Median HH Income': 'Median Household Income ($)',
    'Crime Index': 'Crime Index (AGS)',
    'Transportation barriers crude prevalence (%)': 'Transportation Barriers (Rate per 1,000)',
    'Food insecurity crude prevalence (%)': 'Food Insecurity (Rate per 1,000)',
    'Housing insecurity crude prevalence (%)': 'Housing Insecurity (Rate per 1,000)',
    'Utilities services threat crude prevalence (%)': 'Utility Service Threats (Rate per 1,000)',
    'Social isolation crude prevalence (%)': 'Social Isolation (Rate per 1,000)',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress (Rate per 1,000)',
    'Depression crude prevalence (%)': 'Depression (Rate per 1,000)'
}

def get_label(col):
    return labels.get(col, col.replace(" crude prevalence (%)", " (Rate per 1,000)"))

# --- 3. Statistical Relationships ---
coefficients = {}
for outcome in outcomes:
    coefficients[outcome] = {}
    for driver in drivers:
        X = df[[driver]].fillna(df[driver].mean())
        y = df[outcome].fillna(df[outcome].mean())
        model = LinearRegression().fit(X, y)
        coefficients[outcome][driver] = model.coef_[0]

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health Simulator")

st.title("NJ Municipal Health & Planning Simulator")
st.markdown("""
**Instructions:** Select a municipality. The sliders below are set to the **actual current values** for that area. 
Adjust them to see how changes in social drivers impact health outcomes.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

# Get baseline data for selected muni
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

st.sidebar.header("2. Planning Factors (Actuals)")

# --- SLIDER LOGIC (REPLACED 100% WITH ACTUALS) ---
adjustment_deltas = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    
    # Determine step and range based on the type of data
    if "Income" in driver:
        step_val = 1000.0
        max_val = float(current_val * 2.0)
        format_str = "$%0.0f"
    else:
        step_val = 0.1
        max_val = float(current_val * 2.0) if current_val > 0 else 100.0
        format_str = "%0.1f"

    # The Slider now starts at the ACTUAL baseline value
    new_val = st.sidebar.slider(
        label=get_label(driver),
        min_value=0.0,
        max_value=max_val,
        value=current_val,
        step=step_val,
        format=format_str
    )
    
    # Delta is the difference from the original baseline
    adjustment_deltas[driver] = new_val - current_val

# --- 5. Calculate Predicted Outcomes ---
predicted_values = {}
for outcome in outcomes:
    total_change = sum(adjustment_deltas[d] * coefficients[outcome][d] for d in drivers)
    baseline_val = baseline_data[outcome]
    predicted_values[outcome] = max(0, baseline_val + total_change)

# --- 6. Visualization ---
results = []
for outcome in outcomes:
    results.append({
        "Measure": get_label(outcome).replace(" (Rate per 1,000)", ""),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_values[outcome]
    })
    
results_df = pd.DataFrame(results)

fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Current Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated Scenario', marker_color='royalblue'))

fig.update_layout(
    title=f"Impact Analysis for {selected_muni}",
    yaxis_title="Rate per 1,000",
    barmode='group',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
            label=get_label(outcome).split('(')[0], # Shorten label for space
            value=f"{predicted:.1f}",
            delta=f"{change:.1f}",
            delta_color="inverse" 
        )

st.divider()
st.caption("Note: Calculations are based on linear regression coefficients derived from the full NJ municipal dataset.")
