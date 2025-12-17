import streamlit as st
import numpy as np
import pickle

st.title("🐟 Fish Weight Prediction")

# Load model
with open("fish_poly_model.pkl", "rb") as f:
    model = pickle.load(f)

st.success("Model loaded successfully")

# Inputs (FIXED)
l1 = st.number_input("Length1", value=20.0)
l2 = st.number_input("Length2", value=22.0)
l3 = st.number_input("Length3", value=25.0)
h = st.number_input("Height", value=6.0)
w = st.number_input("Width", value=4.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[l1, l2, l3, h, w]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)
    st.success(f"Predicted Weight: **{prediction[0]:.2f} g**")
