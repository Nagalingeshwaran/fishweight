import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Set Page Config
st.set_page_config(page_title="Fish Weight Predictor", layout="wide", page_icon="🐟")

# --- Load Model and Setup Transformer ---
@st.cache_resource
def load_resources():
    # Load your uploaded pickle model
    with open('fish_poly_model (1).pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Initialize Polynomial transformer (Degree 3)
    poly = PolynomialFeatures(degree=3)
    
    # Pre-fit the transformer to 5 features (Length1, Length2, Length3, Height, Width)
    poly.fit(np.zeros((1, 5))) 
    
    return model, poly

model, poly = load_resources()

# --- Title and Metrics ---
st.title("🐟 Fish Weight Prediction Dashboard")
st.markdown("### Model Validation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", "0.961")
col2.metric("Mean Squared Error", "5555.41")
col3.metric("Polynomial Degree", "3")

st.divider()

# --- App Tabs ---
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (Upload CSV)"])

with tab1:
    st.header("Manual Input")
    st.write("Enter the dimensions for a single fish.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        l1 = st.number_input("Length1 (Vertical)", value=23.2, format="%.2f")
        l2 = st.number_input("Length2 (Diagonal)", value=25.4, format="%.2f")
        l3 = st.number_input("Length3 (Cross)", value=30.0, format="%.2f")
    with col_b:
        h = st.number_input("Height", value=11.52, format="%.4f")
        w = st.number_input("Width", value=4.02, format="%.4f")

    if st.button("Predict Single Weight"):
        input_data = np.array([[l1, l2, l3, h, w]])
        input_poly = poly.transform(input_data)
        prediction = model.predict(input_poly)[0]
        st.success(f"**Predicted Weight:** {max(0, prediction):.2f} grams"
    

    else:
            st.error(f"The CSV must contain these columns: {', '.join(features)}")
