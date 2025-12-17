import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "fish_poly_model.pkl"), "rb") as f:
     model = pickle.load(f)

st.title("🐟 Fish Weight Prediction")

length1 = st.number_input("Length1", min_value=0.0)
length2 = st.number_input("Length2", min_value=0.0)
length3 = st.number_input("Length3", min_value=0.0)
height = st.number_input("Height", min_value=0.0)
width = st.number_input("Width", min_value=0.0)

if st.button("Predict"):
    X = np.array([[length1, length2, length3, height, width]])
    y_poly = poly.transform(X)
    prediction = model.predict(y_poly)
    st.success(f"Predicted Weight: {prediction[0]:.2f} grams")
