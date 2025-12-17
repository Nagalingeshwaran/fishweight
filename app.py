import streamlit as st
import pickle
import numpy as np

# Load model
with open("fish_poly_model.pkl", "rb") as f:
    poly, model = pickle.load(f)

# App title
st.title("🐟 Fish Weight Prediction App")
st.write("Polynomial Regression Model")

# Input fields
length1 = st.number_input("Length1", min_value=0.0)
length2 = st.number_input("Length2", min_value=0.0)
length3 = st.number_input("Length3", min_value=0.0)
height = st.number_input("Height", min_value=0.0)
width = st.number_input("Width", min_value=0.0)

# Predict button
if st.button("Predict Weight"):
    input_data = np.array([[length1, length2, length3, height, width]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)

    st.success(f"Predicted Fish Weight: {prediction[0]:.2f} grams")

