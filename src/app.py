import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib  # Used for loading model files
import os

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "sensor_data.csv"
model_dir = BASE_DIR / "models"

# --- Page Configuration ---
st.set_page_config(page_title="Industrial Equipment Predictive Maintenance Dashboard", layout="wide")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    # Load the "Pro" dataset for visualization
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error(f"Data file {data_path} not found. Please run generate_realistic_data.py first.")
        st.stop()

# --- 2. Load Pre-trained Models ---
@st.cache_resource
def load_trained_models():
    """Load models and feature lists from local files to avoid redundant training"""
    try:
        rf_cls = joblib.load(model_dir / 'rf_model.pkl')
        rf_reg = joblib.load(model_dir / 'rf_reg_model.pkl')
        iso = joblib.load(model_dir / 'iso_model.pkl')
        # Load the feature order recorded during training to prevent errors
        features = joblib.load(model_dir / 'feature_names.pkl')
        return rf_cls, rf_reg, iso, features
    except FileNotFoundError:
        st.error("Model files not found! Please ensure the 'train.py' script has been run.")
        st.stop()

# Initialize data and models
df = load_data()
rf_cls, rf_reg, iso_model, feature_cols = load_trained_models()

# --- 3. Sidebar: Real-time Input Simulator ---
st.sidebar.header("Real-time Sensor Simulator")

def user_input_features():
    # Input fields correspond to training features: 'Vibration', 'Temperature', 'Pressure', 'OperatingHours', 'Vibration_Mean'
    vib = st.sidebar.slider("Current Vibration", 0.0, 1.5, 0.5)
    temp = st.sidebar.slider("Current Temperature", 50.0, 100.0, 70.0)
    pres = st.sidebar.slider("Current Pressure", 50.0, 150.0, 100.0)
    hours = st.sidebar.slider("Total Operating Hours", 0.0, 500.0, 200.0)
    vib_mean = st.sidebar.slider("Rolling Mean Vibration", 0.0, 1.5, 0.5)

    data = {
        'Vibration': vib,
        'Temperature': temp,
        'Pressure': pres,
        'OperatingHours': hours,
        'Vibration_Mean': vib_mean
    }
    # The order here must match feature_cols
    return pd.DataFrame([data])[feature_cols]

input_df = user_input_features()

# --- 4. Main Interface ---
st.title("Smart Factory: Equipment Health Monitoring System (Inference)")
st.markdown("This version loads trained models directly from disk, saving training resources.")

# Row 1: Key Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples", len(df))
c2.metric("Total Failures", int(df['Failure'].sum()))
c3.metric("Feature Dimensions", len(feature_cols))
health_score = max(0, int(100 - (input_df['Vibration'].iloc[0] * 50)))
c4.metric("Real-time Health Score", f"{max(0, int(health_score))}%")

st.divider()

# Row 2: Real-time Prediction Panel
res1, res2 = st.columns(2)
with res1:
    st.subheader("Failure Risk Prediction and RUL")
    # Get failure probability and RUL
    prob = rf_cls.predict_proba(input_df)[0][1]
    predicted_rul = rf_reg.predict(input_df)[0]

    # RUL panel
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Remaining Useful Life (RUL) in hours", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 100], 'color': "red"},
                {'range': [100, 250], 'color': "orange"},
                {'range': [250, 500], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 480  # Assuming a maximum useful life of 500
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prob > 0.5:
        st.error(f"High Risk! Failure Probability: {prob:.2%}")
        if st.button("Create Maintenance Order Now"):
            st.toast("Work order submitted to ERP")
    else:
        st.success(f"Healthy Status (Failure Probability: {prob:.2%})")

with res2:
    st.subheader("Operation Pattern Analysis")
    # Isolation Forest prediction: 1 Normal, -1 Anomaly
    is_anomaly = iso_model.predict(input_df)[0]
    if is_anomaly == -1:
        st.warning("Abnormal pattern detected! Check for sensor drift.")
    else:
        st.info("Normal operation pattern")

st.divider()

# Row 3: Historical Trends
st.subheader("📈 Equipment Degradation Curve (History)")

all_visual_features = ['Vibration', 'Temperature', 'Pressure', 'Vibration_Mean', 'RUL']
selected_features = st.multiselect(
    "Select sensor metrics to monitor:",
    options=all_visual_features,
    default=['Vibration', 'Temperature']  # Default display: Vibration and Temperature
)

if selected_features:
    # Prepare plot data (taking the last 500 data points for better performance/smoothness)
    plot_df = df.tail(500).copy()

    # Use Plotly to draw a multi-line chart
    # Setting y as the list of features selected by the user
    fig_multi = px.line(
        plot_df,
        x='OperatingHours',
        y=selected_features,
        labels={'OperatingHours': 'Operating Hours (Hours)', 'value': 'value', 'variable': 'Sensor Metric'},
        title='Sensor Feature Trends Over Time'
    )

    # Enhance visuals: Mark failure points
    failure_events = plot_df[plot_df['Failure'] == 1]
    if not failure_events.empty:
        for feature in selected_features:
            fig_multi.add_scatter(
                x=failure_events['OperatingHours'],
                y=failure_events[feature],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name=f'{feature} Failure Point',
                showlegend=True
            )

    fig_multi.update_layout(
        hovermode="x unified",  # Display all selected metrics for a specific time point on hover
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_multi, use_container_width=True)
else:
    st.info("Please select at least one metric from the multi-select box above")