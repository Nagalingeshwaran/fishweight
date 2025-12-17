import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Set up the app title and description
st.title("🐟 Fish Prediction App")
st.write("Enter the input feature below to get a prediction from the polynomial regression model.")

# Load the model
@st.cache_resource
def load_model():
    with open('fish_poly_model (1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# User Input
# Replace 'Input Feature' with the actual name (e.g., 'Age' or 'Length')
user_input = st.number_input("Enter the input value:", min_value=0.0, value=1.0, step=0.1)

if st.button("Predict"):
    # 1. Reshape input for sklearn
    X_input = np.array([[user_input]])
    
    # 2. Transform to Polynomial Features
    # Note: adjust 'degree' if your model used something other than 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_input)
    
    # 3. Make Prediction
    prediction = model.predict(X_poly)
    
    # 4. Display Result
    st.success(f"The predicted value is: {prediction[0]:.2f}")

