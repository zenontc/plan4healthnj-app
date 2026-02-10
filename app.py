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

st.sidebar.header("2. Adjust Factors")
st.sidebar.markdown("Use sliders to show changes. Values are the percentage of the population affected.")

if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

slider_vals = {}
for driver in drivers:
    current_val = float(baseline_data[driver])
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.5), 1000.0
        fmt = "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f"
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1
        fmt = "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    slider_vals[driver] = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v,
        max_value=max_v,
        key=f"slider_{driver}",
        step=step,
        format=fmt
    )

# --- 6. Calculation Logic ---
# Logic: If all factors except Income are 0, or all factors are 0, results must be 0.
# We use a "Presence Factor" to scale the results.
non_income_drivers = [d for d in drivers if "Income" not in d]
current_sum = sum([slider_vals[d] for d in non_income_drivers])
baseline_sum = sum([baseline_data[d] for d in non_income_drivers])

# Ratio of current environment vs baseline environment
presence_ratio = (current_sum / baseline_sum) if baseline_sum > 0 else 0

predicted_values = {}
DAMPING_FACTOR = 0.25 # More conservative

for outcome in outcomes:
    total_delta = 0
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        
        # Apply specific logic boosts
        weight = 1.0
        if driver == 'Physical inactivity crude prevalence (%)' and outcome in ['Obesity crude prevalence (%)', 'Diabetes crude prevalence (%)']:
            weight = 1.8
        if driver == 'Social isolation crude prevalence (%)' and outcome == 'Depression crude prevalence (%)':
            weight = 2.0
            
        delta = (slider_vals[driver] - baseline_data[driver]) * weight
        total_delta += delta * coeff

    # Composite buffer for Poor Health (makes it harder to move)
    if outcome == 'Fair or poor health crude prevalence (%)':
        total_delta *= 0.5 

    # Calculate predicted value
    raw_simulated = baseline_data[outcome] + (total_delta * DAMPING_FACTOR)
    
    # Apply Presence Scaling: If factors are reduced to 0, outcome reduces to 0.
    # We use a power function so it stays conservative until the very end.
    final_val = raw_simulated * (presence_ratio ** 0.5)
    
    predicted_values[outcome] = max(0, final_val)

# --- 7. Visualizations ---
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
fig.update_layout(title=f"Projected Health Outcomes: {selected_muni}", yaxis_title="Prevalence (%)", barmode='group', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- 8. Metrics Table ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))
for i, outcome in enumerate(outcomes):
    base, pred = baseline_data[outcome], predicted_values[outcome]
    diff = pred - base
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None # Start grayed out/no arrow for tiny changes
    with cols[i]:
        st.metric(label=get_clean_label(outcome), value=f"{pred:.1f}%", delta=delta_val, delta_color="inverse")

st.markdown("---")
st.caption("*Note: This model uses a conservative decay function. Outcomes are proportionally tied to environmental drivers; if all environmental factors reach zero, the model projects a zero-prevalence scenario.*")
