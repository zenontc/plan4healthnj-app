import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('NJ_Municipal_Health_Data.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- 2. Define Drivers and Outcomes (Specific Order) ---
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
    'Frequnt physical distress crude prevalence (%)', # Typo from CSV
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# --- 3. Helper Functions ---
def get_clean_label(col):
    mapping = {
        'Fair or poor health crude prevalence (%)': 'Poor Health (%)',
        'Frequnt physical distress crude prevalence (%)': 'Physical Distress (%)',
        'Median HH Income': 'Median HH Income ($)'
    }
    if col in mapping:
        return mapping[col]
    return col.replace(" crude prevalence (%)", " (%)")

# --- 4. Regression & Interdependency Logic ---
@st.cache_resource
def get_coefficients(_df, _drivers, _outcomes):
    coeffs = {}
    for outcome in _outcomes:
        coeffs[outcome] = {}
        for driver in _drivers:
            temp_df = _df[[driver, outcome]].dropna()
            X = temp_df[[driver]]
            y = temp_df[outcome]
            model = LinearRegression().fit(X, y)
            coeffs[outcome][driver] = model.coef_[0]
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 5. UI Layout ---
st.set_page_config(layout="wide", page_title="NJ Health & Planning Tool")
st.title("NJ Municipal Health & Planning Simulator")
st.markdown("""
This tool simulates how changes in the **built environment** and **social factors** (drivers) 
might impact **public health outcomes** in New Jersey municipalities.
""")

# Sidebar: Select Municipality
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())

# Reset logic for municipality change
if 'last_muni' not in st.session_state:
    st.session_state.last_muni = muni_list[0]

selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)

# If municipality changes, reset all sliders in session state
if selected_muni != st.session_state.last_muni:
    for d in drivers:
        if f"slider_{d}" in st.session_state:
            del st.session_state[f"slider_{d}"]
    st.session_state.last_muni = selected_muni

baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# Sidebar: Drivers
st.sidebar.header("2. Adjust Planning Factors")
st.sidebar.markdown("Use sliders to simulate improvements or decline.")

# Reset Button
if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

adjustment_deltas = {}

for driver in drivers:
    current_val = float(baseline_data[driver])
    
    # Dynamic range logic
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.2), 1000.0
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), current_val * 2), 0.1

    # Initialize slider in session state if not there
    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = current_val

    val = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v,
        max_value=max_v,
        key=f"slider_{driver}",
        step=step,
        format="$%.0f" if "Income" in driver else "%.1f%%"
    )
    adjustment_deltas[driver] = val - current_val

# --- 6. Inter-Driver Logic (Sliders affecting one another) ---
# For instance, if Transportation barriers drop, Food Insecurity and Social Isolation also drop.
# We modify the deltas used for the final outcome calculation.
final_deltas = adjustment_deltas.copy()

# Logic: Transportation impacts Food Insecurity (30% weight) and Social Isolation (20% weight)
final_deltas['Food insecurity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.3
final_deltas['Social isolation crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.2
# Logic: Income impacts Housing and Food Insecurity (scaled appropriately)
income_delta_scaled = adjustment_deltas['Median HH Income'] / 10000 
final_deltas['Housing insecurity crude prevalence (%)'] -= income_delta_scaled * 0.5
final_deltas['Food insecurity crude prevalence (%)'] -= income_delta_scaled * 0.3

# --- 7. Calculate Outcomes with Damping ---
DAMPING_FACTOR = 0.4  # Makes the model more conservative
predicted_values = {}

for outcome in outcomes:
    total_change = 0
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        total_change += final_deltas[driver] * coeff
    
    # Apply damping and ensure no negative prevalence
    total_change *= DAMPING_FACTOR
    predicted_values[outcome] = max(0, baseline_data[outcome] + total_change)

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
    
    # Formatting for 0.0% delta
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.01 else None
    
    with cols[i]:
        st.metric(
            label=get_clean_label(outcome),
            value=f"{pred:.1f}%",
            delta=delta_val,
            delta_color="inverse"
        )

st.markdown("---")
st.caption("*Note: This simulator uses linear regression with a damping factor for conservative estimation. Driver interdependencies (e.g., Transportation impacting Food Insecurity) are included in the logic.*")
