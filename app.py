import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Fish Weight Predictor", layout="wide")

st.title("🐟 Fish Weight Prediction App")
st.write("""
This app uses a Polynomial Linear Regression model to predict the weight of a fish 
based on 26 distinct features.
""")

# Load the model
@st.cache_resource
def load_model():
    # Ensure fish_poly_model.pkl is in the same directory
    return joblib.load("fish_poly_model.pkl")

try:
    model = load_model()
    
    st.sidebar.header("Input Fish Metrics")
    st.sidebar.info("The model requires 26 numerical features (polynomial features).")

    # Create 26 input fields dynamically
    inputs = []
    cols = st.columns(4) # Distribute inputs in 4 columns for better UI
    
    for i in range(26):
        with cols[i % 4]:
            val = st.number_input(f"Feature {i+1}", value=0.0, step=0.1, format="%.2f")
            inputs.append(val)

    # Prediction logic
    if st.button("Predict Weight"):
        # Reshape input for sklearn
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        
        st.success(f"### Predicted Weight: {prediction[0]:.2f} units")
        
        # Display the input data for reference
        with st.expander("View Input Vector"):
            st.write(input_array)

except FileNotFoundError:
    st.error("Error: 'fish_poly_model.pkl' not found. Please ensure the file is in the app directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
