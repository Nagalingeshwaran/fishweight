import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Fish Weight Prediction", layout="centered")

st.title("🐟 Fish Weight Prediction")
st.write("Predict fish weight using a **Polynomial Regression model**")

# Load trained model
with open("fish_poly_model.pkl", "rb") as f:
    poly, model = pickle.load(f)

st.success("✅ Model loaded successfully")

# User input section
st.subheader("Enter Fish Measurements")

l1 = st.number_input("Length1 (cm)", min_value=0.0, max_value=100.0, value=20.0)
l2 = st.number_input("Length2 (cm)", min_value=0.0, max_value=100.0, value=22.0)
l3 = st.number_input("Length3 (cm)", min_value=0.0, max_value=100.0, value=25.0)
h  = st.number_input("Height (cm)",  min_value=0.0, max_value=50.0,  value=6.0)
w  = st.number_input("Width (cm)",   min_value=0.0, max_value=30.0,  value=4.0)

# Prediction
if st.button("Predict Weight"):
    input_data = np.array([[l1, l2, l3, h, w]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)

    st.success(f"🐟 Predicted Fish Weight: **{prediction[0]:.2f} grams**")
