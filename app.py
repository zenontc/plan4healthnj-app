import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
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

# --- 2. Define Drivers (Inputs) and Outcomes (Outputs) ---
# Ordered as requested
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
    'Fair or poor health crude prevalence (%)' # Will rename to "Poor Health" in UI
]

# --- 3. Calculate Statistical Relationships ---
# We use a damping factor (0.4) to make the model more conservative
SENSITIVITY_DAMPING = 0.4 

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
                # Apply damping to keep results conservative
                coeffs[outcome][driver] = model.coef_[0] * SENSITIVITY_DAMPING
            else:
                coeffs[outcome][driver] = 0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 4. Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health & Planning Tool")

st.title("NJ Municipal Planning and Public Health Tool")
st.markdown("""
This tool models how changes in factors that can be impacted through Planning 
might impact **public health outcomes** in New Jersey municipalities.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# --- RESET LOGIC ---
if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

def reset_sliders():
    st.session_state.reset_key += 1

st.sidebar.button("Reset Sliders to Baseline", on_click=reset_sliders)

st.sidebar.header("2. Adjust Planning Factors")
st.sidebar.markdown("Use sliders to see improvements or decline in outcomes.")

slider_vals = {}
for driver in drivers:
    current_val = float(baseline_data[driver])
    
    # Range logic
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max()), 1000.0
        fmt = "$%.0f"
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f%%"

    # Unique key with reset trigger
    slider_vals[driver] = st.sidebar.slider(
        f"{driver}",
        min_value=min_v,
        max_value=max_v,
        value=current_val,
        step=step,
        format=fmt,
        key=f"{driver}_{st.session_state.reset_key}"
    )

# --- 5. INTER-SLIDER DEPENDENCY LOGIC ---
# Changes in one factor influence others before calculating health outcomes
adj_deltas = {d: slider_vals[d] - baseline_data[d] for d in drivers}

# Example Dependencies:
# 1. Transportation improvement reduces Food Insecurity delta
adj_deltas['Food insecurity crude prevalence (%)'] += (adj_deltas['Transportation barriers crude prevalence (%)'] * 0.2)
# 2. Income increase reduces Housing and Food insecurity deltas
if "Median HH Income" in adj_deltas:
    income_effect = adj_deltas['Median HH Income'] / 10000  # Per 10k change
    adj_deltas['Housing insecurity crude prevalence (%)'] -= (income_effect * 0.5)
    adj_deltas['Food insecurity crude prevalence (%)'] -= (income_effect * 0.3)

# --- 6. Calculate Predicted Outcomes ---
predicted_values = {}
for outcome in outcomes:
    total_change = sum(adj_deltas[d] * coefficients[outcome][d] for d in drivers)
    predicted_val = max(0, baseline_data[outcome] + total_change)
    predicted_values[outcome] = predicted_val

# --- 7. Visualization ---
# Rename "Fair or poor health" to "Poor Health" for results
def clean_label(text):
    text = text.replace(" crude prevalence (%)", "")
    if "Fair or poor health" in text:
        return "Poor Health"
    return text

results = []
for outcome in outcomes:
    results.append({
        "Measure": clean_label(outcome),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted_values[outcome]
    })
    
results_df = pd.DataFrame(results)

fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Current Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated Scenario', marker_color='royalblue'))

fig.update_layout(
    title=f"Projected Health Outcomes for {selected_muni}",
    yaxis_title="Prevalence (%)",
    barmode='group',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- 8. Detailed Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))

for i, outcome in enumerate(outcomes):
    baseline = baseline_data[outcome]
    predicted = predicted_values[outcome]
    change = predicted - baseline
    
    # Delta Formatting: Gray and no arrow if change is 0
    d_color = "inverse"
    if round(change, 1) == 0.0:
        d_color = "off"
    
    with cols[i]:
        st.metric(
            label=clean_label(outcome),
            value=f"{predicted:.1f}%",
            delta=f"{change:.1f}%" if d_color != "off" else "0.0%",
            delta_color=d_color
        )

st.markdown("---")
st.caption("*Note: This model uses simple linear regression coefficients derived from NJ municipal data. It assumes additive effects and includes conservative damping factors for planning simulation purposes only.*")


