import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
try:
    model = joblib.load('random_forest_model_compressed.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error(f"Failed to load model or encoder: {e}")
    st.stop()

st.title("Nano FET Id Classification App (Random Forest)")

st.markdown("Enter the following parameters to predict the Drain Current Class (Id):")

# User inputs
vg_input = st.number_input("Vg (in V)", min_value=0.0, max_value=2.0, step=0.1)
gate_input = st.number_input("Gate Length (nm)", min_value=1.0, max_value=10.0, step=0.1)
tox_input = st.number_input("Tox", min_value=1.0, max_value=10.0, step=0.1)
side_input = st.number_input("Number of Sides", min_value=3, max_value=8, step=1)
trap_input = st.selectbox("Trap Charge Present?", ["No", "Yes"])
trap_input_val = 1.0 if trap_input == "Yes" else 0.0
work_function_input = st.number_input("Work Function (eV)", min_value=4.3, max_value=4.9, step=0.1)

if st.button("Predict Id"):
    try:
        custom_input = np.array([[vg_input, gate_input, tox_input, side_input, trap_input_val, work_function_input]])
        predicted_class_index = model.predict(custom_input)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
        st.success(f"Predicted Id Class: {predicted_label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
