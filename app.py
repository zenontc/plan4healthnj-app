import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('NJ_Municipal_Health_Data.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("🚨 **File Not Found:** Ensure 'NJ_Municipal_Health_Data.csv' is in your directory.")
    st.stop()

# --- 2. Define Ordered Drivers and Outcomes ---
# Reordered as requested
drivers = [
    'Median HH Income',
    'Crime Index',
    'Transportation barriers crude prevalence (%)',
    'Food insecurity crude prevalence (%)',
    'Housing insecurity crude prevalence (%)',
    'Utilities services threat crude prevalence (%)',
    'Social isolation crude prevalence (%)'
]

# Reordered as requested: Distress and Depression after Diabetes
outcomes = [
    'Physical inactivity crude prevalence (%)',
    'Obesity crude prevalence (%)',
    'Diabetes crude prevalence (%)',
    'Frequnt physical distress crude prevalence (%)', 
    'Depression crude prevalence (%)',
    'Frequent mental distress crude prevalence (%)',
    'Fair or poor health crude prevalence (%)'
]

# Label mapping for "Poor Health" rename
labels = {
    'Fair or poor health crude prevalence (%)': 'Poor Health',
    'Frequnt physical distress crude prevalence (%)': 'Frequent Physical Distress',
    'Median HH Income': 'Median HH Income ($)'
}

def get_label(col):
    return labels.get(col, col.replace(" crude prevalence (%)", ""))

# --- 3. Calculate Statistical Relationships ---
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
                coeffs[outcome][driver] = 0
    return coeffs

coefficients = get_coefficients(df, drivers, outcomes)

# --- 4. UI Layout & Session State ---
st.set_page_config(layout="wide", page_title="NJ Health Simulator")
st.title("NJ Municipal Health & Planning Simulator")

# Sidebar: Municipality Selection
st.sidebar.header("1. Select Municipality")
muni_list = sorted(df['Municipality and County'].unique())
selected_muni = st.sidebar.selectbox("Choose a Municipality:", muni_list)
baseline_data = df[df['Municipality and County'] == selected_muni].iloc[0]

# --- RESET LOGIC ---
if "last_muni" not in st.session_state or st.session_state.last_muni != selected_muni:
    st.session_state.last_muni = selected_muni
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

def reset_sliders():
    for d in drivers:
        st.session_state[f"slider_{d}"] = float(baseline_data[d])

st.sidebar.button("Reset Sliders to Baseline", on_click=reset_sliders)

st.sidebar.header("2. Planning Factors (Actuals)")
raw_deltas = {}

# Display Sliders in requested order
for driver in drivers:
    current_val = float(baseline_data[driver])
    max_range = float(current_val * 2.0) if current_val > 0 else 100.0
    step = 500.0 if "Income" in driver else 0.1
    fmt = "$%0.0f" if "Income" in driver else "%0.1f"

    st.sidebar.markdown(f"**{get_label(driver)}**")
    val = st.sidebar.slider(
        label=driver,
        min_value=0.0,
        max_value=max_range,
        key=f"slider_{driver}",
        step=step,
        format=fmt,
        label_visibility="collapsed"
    )
    raw_deltas[driver] = val - current_val

# --- 5. MODEL LOGIC: Dependencies & Dampening ---

# A. Inter-Driver Dependencies (The "Ripple Effect")
# Example: Reducing Transportation barriers reduces Food Insecurity by 20% of that delta
adj_deltas = raw_deltas.copy()
adj_deltas['Food insecurity crude prevalence (%)'] += raw_deltas['Transportation barriers crude prevalence (%)'] * 0.20
adj_deltas['Crime Index'] += (raw_deltas['Median HH Income'] / 1000) * -0.05 # Income affects crime

# B. Conservative Dampening Factor
# This prevents extreme swings (0.4 means only 40% of the statistical impact is applied)
DAMPENING = 0.4 

results = []
for outcome in outcomes:
    # Weighted impact calculation
    impact = sum(adj_deltas[d] * coefficients[outcome][d] for d in drivers)
    predicted = max(0, baseline_data[outcome] + (impact * DAMPENING))
    
    results.append({
        "Measure": get_label(outcome),
        "Baseline": baseline_data[outcome],
        "Simulated": predicted,
        "Change": predicted - baseline_data[outcome]
    })

results_df = pd.DataFrame(results)

# --- 6. Visualizations ---
fig = go.Figure()
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Baseline"], name='Baseline', marker_color='lightslategray'))
fig.add_trace(go.Bar(x=results_df["Measure"], y=results_df["Simulated"], name='Simulated', marker_color='royalblue'))

fig.update_layout(
    title=f"Projected Health Outcomes for {selected_muni}",
    yaxis_title="Crude Prevalence",
    barmode='group',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- 7. Detailed Metrics (Gray 0.0% Logic) ---
st.subheader("Detailed Projections")
cols = st.columns(len(outcomes))

for i, row in results_df.iterrows():
    with cols[i]:
        # Logic for gray 0.0 and no arrow
        if abs(row['Change']) < 0.01:
            st.metric(label=row['Measure'], value=f"{row['Simulated']:.1f}", delta=None)
        else:
            st.metric(
                label=row['Measure'], 
                value=f"{row['Simulated']:.1f}", 
                delta=f"{row['Change']:.1f}", 
                delta_color="inverse"
            )

st.divider()
st.caption("Note: Impact is dampened for conservative estimation. 'Poor Health' combines Fair/Poor categories.")
