import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
def load_model():
    with open('fish_poly_model (1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Header
st.title("🐟 Fish Weight Predictor")
st.write("Enter the fish measurements below to predict its weight in grams.")

# Layout with columns for input
col1, col2 = st.columns(2)

with col1:
    length1 = st.number_input("Vertical Length (cm)", min_value=0.0, value=20.0)
    length2 = st.number_input("Diagonal Length (cm)", min_value=0.0, value=22.0)
    length3 = st.number_input("Cross Length (cm)", min_value=0.0, value=25.0)

with col2:
    height = st.number_input("Height (cm)", min_value=0.0, value=10.0)
    width = st.number_input("Diagonal Width (cm)", min_value=0.0, value=4.0)

# Prediction Logic
if st.button("Predict Weight"):
    # The model expects 18 features (common for Polynomial degree 2 with 5 inputs)
    # This part depends on how you processed your features during training.
    # For now, we simulate the input vector structure:
    try:
        # We need to create the polynomial features manually if you didn't save the 'poly' object
        # This is a standard degree 2 expansion for Fish Market data:
        raw_inputs = [length1, length2, length3, height, width]
        
        # If your model was trained using Scikit-Learn's PolynomialFeatures(degree=2),
        # it expects a specific sequence of squared and interaction terms.
        # Here we attempt to pass the data:
        
        # Note: If your model throws an error about 'feature names', we use a dummy dataframe
        features = np.array([raw_inputs]) 
        
        # IMPORTANT: Since your model has 18 features, it's likely expecting 
        # the output of PolynomialFeatures(degree=2, include_bias=False).
        # We will manually expand them:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        expanded_features = poly.fit_transform(features)

        prediction = model.predict(expanded_features)
        
        # Display Result
        st.success(f"The estimated weight of the fish is: {prediction[0]:.2f} grams")
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.info("Technical Note: The model expects 18 input features. Ensure the polynomial transformation matches your training phase.")

# Sidebar info
st.sidebar.info("This model uses a Linear Regression algorithm with Polynomial features.")
