# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model (trained without 'tox')
model = load_model('cnn_model_no_scaling.keras')

st.title("Nano FET Id Prediction App (No Scaling)")

st.markdown("Enter the following parameters to predict the Drain Current (Id):")

# User inputs
vg_input = st.number_input("Vg (in V)", min_value=0.0, max_value=5.0, step=0.1, format="%.2f")
gate_input = st.number_input("Gate Length (nm)", min_value=1.0, max_value=100.0, step=0.1, format="%.2f")
tox_input = st.number_input("Tox", min_value=0.1, max_value=10.0, step=0.1, format="%.2f")  # Not used
side_input = st.number_input("Number of Sides", min_value=3, max_value=10, step=1)
trap_input = st.selectbox("Trap Charge Present?", ["No", "Yes"])
trap_input_val = 1.0 if trap_input == "Yes" else 0.0
work_function_input = st.number_input("Work Function (eV)", min_value=4.3, max_value=5.0, step=0.1, format="%.2f")

if st.button("Predict Id"):
    # tox_input is ignored
    custom_input = np.array([[vg_input, gate_input, side_input, trap_input_val, work_function_input]])
    custom_input_reshaped = custom_input.reshape(1, 5, 1)

    predicted_id = model.predict(custom_input_reshaped, verbose=0)[0][0]

    st.success(f"Predicted Id: {predicted_id:.4e} A")