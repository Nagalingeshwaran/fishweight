import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(page_title="Fish Weight Predictor", layout="centered")

# --- Load the Model and Preprocessor ---
@st.cache_resource
def load_model_and_poly():
    # Load the pre-trained linear regression model
    with open('fish_poly_model (1).pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Recreate the polynomial transformer used in the notebook (degree 3)
    poly = PolynomialFeatures(degree=2)
    
    # The transformer needs to be "fitted" to the structure of the data. 
    # Based on the notebook, it was trained on 5 features.
    # We simulate a fit with dummy data to prepare the transformer.
    dummy_data = np.zeros((1, 5)) 
    poly.fit(dummy_data)
    
    return model, poly

try:
    model, poly = load_model_and_poly()

    st.title("🐟 Fish Weight Prediction")
    st.write("Enter the physical measurements of the fish to predict its weight in grams.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Fish Measurements")
    
    # Input fields based on the features defined in the notebook 
    l1 = st.sidebar.number_input("Length1 (Vertical)", min_value=0.0, value=23.2, help="Vertical length in cm")
    l2 = st.sidebar.number_input("Length2 (Diagonal)", min_value=0.0, value=25.4, help="Diagonal length in cm")
    l3 = st.sidebar.number_input("Length3 (Cross)", min_value=0.0, value=30.0, help="Cross length in cm")
    height = st.sidebar.number_input("Height", min_value=0.0, value=11.52, help="Height in cm")
    width = st.sidebar.number_input("Width", min_value=0.0, value=4.02, help="Diagonal width in cm")

    # --- Prediction ---
    if st.button("Predict Weight"):
        # Organize inputs into the correct format 
        input_data = np.array([[l1, l2, l3, height, width]])
        
        # Transform inputs to polynomial features 
        input_poly = poly.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_poly)
        
        # Display results
        result = max(0, prediction[0]) # Ensure we don't display negative weight
        st.success(f"### Estimated Weight: {result:.2f} grams")
        
        # Metrics context from the notebook 
        st.info("Note: This prediction is based on a Polynomial Regression model (Degree 3) "
                "which achieved an R² score of 0.961 during training.")

except FileNotFoundError:
    st.error("Error: 'fish_poly_model.pkl' not found. Please ensure the model file is in the same directory.")
