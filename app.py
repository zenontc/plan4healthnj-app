import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('NJ_Municipal_Health_Data.csv')
        df.columns = df.columns.str.strip()  # Remove hidden spaces
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("🚨 **File Not Found:** Ensure 'NJ_Municipal_Health_Data.csv' is in your GitHub repo or local folder.")
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

# Quick check for matching column names
missing_cols = [c for c in drivers + outcomes if c not in df.columns]
if missing_cols:
    st.error(f"🚨 **Column Mismatch:** {missing_cols}")
    st.stop()

# --- 3. Calculate Statistical Relationships ---
@st.cache_resource
def get_coefficients(_df, _drivers, _outcomes):
    coeffs = {}
    for outcome in _outcomes:
        coeffs[outcome] = {}
        for driver in _drivers:
            # Drop empty rows to prevent fit errors
            temp_df = _df[[driver, outcome]].dropna()
            if not temp_df.empty:
                X = temp_df[[driver]]
                y = temp_df[outcome]
                # Syntax corrected here: fit(X, y)
                model = LinearRegression().fit(X, y)
                coeffs[outcome][driver] = model.coef_[0]
            else:
                coeffs[outcome][driver] = 0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 4. UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health Simulator")
st.title("NJ Municipal Health & Planning Simulator")

# Sidebar: Selection
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

st.sidebar.header("2. Planning Factors (Actuals)")
adjustment_deltas = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    
    # Scale behavior: 0 to 2x Baseline (Actual values)
    max_range = float(current_val * 2.0) if current_val > 0 else 100.0
    step = 500.0 if "Income" in driver else 0.1
    fmt = "$%0.0f" if "Income" in driver else "%0.1f"

    st.sidebar.markdown(f"**{driver.replace(' crude prevalence (%)', '')}**")
    simulated_val = st.sidebar.slider(
        label=f"Adjust {driver}",
        min_value=0.0,
        max_value=max_range,
        value=current_val,
        step=step,
        format=fmt,
        label_visibility="collapsed"
    )
    # The Delta: (User Input - Original Value)
    adjustment_deltas[driver] = simulated_val - current_val

# --- 5. Calculate and Visualize ---
results = []
for outcome in outcomes:
    # Change = Sum of (Delta * Coefficient) for all drivers
    total_change = sum(adjustment_deltas[d] * coefficients[outcome][d] for d in drivers)
    predicted = max(0, baseline_data[outcome] + total_change)
    results.append({
        "Measure": outcome.replace(" crude prevalence (%)", ""),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted
    })

results_df = pd.DataFrame(results)

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))
fig.update_layout(title=f"Projections for {selected_muni}", yaxis_title="Crude Prevalence", barmode='group')
st.plotly_chart(fig, use_container_width=True)

# Metrics
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))
for i, row in results_df.iterrows():
    change = row['Simulated'] - row['Baseline']
    with cols[i]:
        st.metric(
            label=row['Measure'], 
            value=f"{row['Simulated']:.1f}", 
            delta=f"{change:.1f}", 
            delta_color="inverse"
        )
