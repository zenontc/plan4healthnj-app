import streamlit as st
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Health Planning Tool", layout="wide")

# Header Section
st.title("🏥 Health Service Planning Factor Tool")
st.markdown("""
Reverted to **Actual Values** and **Crude Prevalence** (per 1,000 pop). 
Use the sidebar to adjust your catchment population and disease frequency.
""")

st.divider()

# --- SIDEBAR: PLANNING FACTORS ---
st.sidebar.header("Step 1: Planning Factors")

# 1. Base Population (Actual Value)
total_population = st.sidebar.number_input(
    "Catchment Population (Total)", 
    min_value=1000, 
    value=50000, 
    step=1000,
    help="The total number of people in your service area."
)

# 2. Crude Prevalence (Cases per 1,000)
crude_prevalence = st.sidebar.slider(
    "Crude Prevalence (per 1,000 people)", 
    min_value=0.1, 
    max_value=100.0, 
    value=15.0,
    help="The total number of existing cases for every 1,000 people in the population."
)

# --- CALCULATION LOGIC ---
# Planned Cases = (Crude Prevalence / 1,000) * Total Population
planned_cases = int((crude_prevalence / 1000) * total_population)

# --- DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Catchment Population", value=f"{total_population:,}")

with col2:
    st.metric(label="Prevalence Rate", value=f"{crude_prevalence}/1,000")

with col3:
    st.metric(label="Estimated Total Cases", value=f"{planned_cases:,}")

st.divider()

# --- DATA BREAKDOWN ---
st.subheader("📊 Planning Breakdown")

# Breakdown of cases by demographic or severity
data = {
    "Category": ["Mild / Routine", "Moderate / Urgent", "Severe / Hospitalization"],
    "Factor of Total": [0.70, 0.20, 0.10], 
}

df = pd.DataFrame(data)
df["Actual Expected Count"] = (df["Factor of Total"] * planned_cases).astype(int)

# Formatting for display
st.table(df[["Category", "Actual Expected Count"]])

st.info(f"**Note:** This model assumes a crude prevalence of {crude_prevalence} per 1,000. "
        f"For a population of {total_population:,}, your facility should prepare for {planned_cases:,} cases.")
