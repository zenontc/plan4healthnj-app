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
st.sidebar.markdown("Use sliders to show changes. Physical Inactivity to Social Isolation factors are the percentage of the population over 18 affected by the issue.")

if st.sidebar.button("Reset to Baseline"):
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

current_slider_values = {}
adjustment_deltas = {}

for driver in drivers:
    baseline_val = float(baseline_data[driver])
    
    if "Income" in driver:
        min_v, max_v, step = 0.0, float(df[driver].max() * 1.2), 1000.0
        fmt = "$%.0f"
    elif "Crime Index" in driver:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), baseline_val * 2), 0.1
        fmt = "%.1f"
    else:
        min_v, max_v, step = 0.0, max(float(df[driver].max()), baseline_val * 2), 0.1
        fmt = "%.1f%%"

    if f"slider_{driver}" not in st.session_state:
        st.session_state[f"slider_{driver}"] = baseline_val

    val = st.sidebar.slider(
        get_clean_label(driver),
        min_value=min_v,
        max_value=max_v,
        key=f"slider_{driver}",
        step=step,
        format=fmt
    )
    current_slider_values[driver] = val
    adjustment_deltas[driver] = val - baseline_val

# --- 6. Inter-Driver Logic (Ripple Effects) ---
final_deltas = adjustment_deltas.copy()

# Transportation impacts
final_deltas['Food insecurity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.3
final_deltas['Physical inactivity crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.3
final_deltas['Social isolation crude prevalence (%)'] += adjustment_deltas['Transportation barriers crude prevalence (%)'] * 0.2

# Income impacts
income_delta_scaled = adjustment_deltas['Median HH Income'] / 10000 
final_deltas['Housing insecurity crude prevalence (%)'] -= income_delta_scaled * 0.5
final_deltas['Food insecurity crude prevalence (%)'] -= income_delta_scaled * 0.3

# Crime impacts
crime_delta_scaled = adjustment_deltas['Crime Index'] / 10
final_deltas['Physical inactivity crude prevalence (%)'] += crime_delta_scaled * 0.2

# --- 7. Calculate Outcomes with Enhanced Logic ---
DAMPING_FACTOR = 0.25 # More conservative base
predicted_values = {}

# Check if all social/environmental factors are zero (ignoring income)
all_zero_except_income = all(current_slider_values[d] == 0 for d in drivers if d != 'Median HH Income')

for outcome in outcomes:
    total_change = 0
    for driver in drivers:
        coeff = coefficients[outcome][driver]
        
        # Boost specific relationships
        weight = 1.0
        if driver == 'Physical inactivity crude prevalence (%)' and outcome in ['Obesity crude prevalence (%)', 'Diabetes crude prevalence (%)']:
            weight = 1.8
        if driver == 'Social isolation crude prevalence (%)' and outcome == 'Depression crude prevalence (%)':
            weight = 1.8
            
        total_change += (final_deltas[driver] * coeff * weight)
    
    # Poor Health is a composite; it moves slower than individual diseases
    if outcome == 'Fair or poor health crude prevalence (%)':
        effective_damping = DAMPING_FACTOR * 0.6
    else:
        effective_damping = DAMPING_FACTOR

    # Special logic: if all factors are zero, outcomes should drop to zero
    if all_zero_except_income:
        predicted_values[outcome] = 0.0
    else:
        # Standard calculation
        predicted = baseline_data[outcome] + (total_change * effective_damping)
        # Ensure it doesn't go to zero unless factors are zero
        predicted_values[outcome] = max(0.1, predicted) if baseline_data[outcome] > 0 else 0.0

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
    
    # Logic to handle the gray 0.0% delta
    delta_val = f"{diff:.1f}%" if abs(diff) > 0.05 else None
    
    with cols[i]:
        st.metric(
            label=get_clean_label(outcome),
            value=f"{pred:.1f}%",
            delta=delta_val,
            delta_color="inverse"
        )

st.markdown("---")
st.caption("*Note: This simulator uses a conservative damping factor and weighted interdependencies. Health outcomes are programmed to reach zero only when all social determinants are addressed.*")
