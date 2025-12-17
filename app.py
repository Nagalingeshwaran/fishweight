import streamlit as st
import numpy as np
import pickle

# App title
st.title("🐟 Fish Weight Prediction using Polynomial Regression")

st.write("""
This app predicts **Fish Weight** using a **Pre-trained Polynomial Regression Model**.
""")

# Load model
with open("fish_poly_model.pkl", "rb") as f:
     model = pickle.load(f)

st.success("Model loaded successfully!")

# Input section
st.subheader("Enter Fish Measurements")

l1 = st.number_input("Length1", value=20.0)
l2 = st.number_input("Length2", value=22.0)
l3 = st.number_input("Length3", value=25.0)
h = st.number_input("Height", value=6.0)
w = st.number_input("Width", value=4.0)

# Prediction
if st.button("Predict Weight"):
    input_data = np.array([[l1, l2, l3, h, w]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)
    st.success(f"Predicted Fish Weight: **{prediction[0]:.2f} grams**")
