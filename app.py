import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(page_title="Fish Weight Predictor", page_icon="🐟")

# --- Load the Model and Setup Transformer ---
@st.cache_resource
def load_resources():
    # Load your uploaded pickle model
    with open('fish_poly_model (1).pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Initialize the same Polynomial transformer used in training (Degree 3)
    poly = PolynomialFeatures(degree=3)
    
    # We need to fit the transformer once so it knows the input shape (5 features)
    # Features: Length1, Length2, Length3, Height, Width
    poly.fit(np.zeros((1, 5))) 
    
    return model, poly

model, poly = load_resources()

# --- User Interface ---
st.title("🐟 Fish Weight Prediction App")
st.markdown("""
This app predicts the weight of a fish based on its physical measurements 
using a **Polynomial Regression Model**.
""")

st.sidebar.header("Input Fish Measurements")

def user_input_features():
    l1 = st.sidebar.number_input("Vertical Length (Length1)", value=23.2)
    l2 = st.sidebar.number_input("Diagonal Length (Length2)", value=25.4)
    l3 = st.sidebar.number_input("Cross Length (Length3)", value=30.0)
    height = st.sidebar.number_input("Height", value=11.5)
    width = st.sidebar.number_input("Diagonal Width", value=4.0)
    
    data = {'Length1': l1,
            'Length2': l2,
            'Length3': l3,
            'Height': height,
            'Width': width}
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Display the user inputs
st.subheader("Selected Measurements")
st.write(df_input)

# --- Prediction Logic ---
if st.button("Predict Weight"):
    # 1. Transform the input into polynomial features
    input_poly = poly.transform(df_input)
    
    # 2. Predict using the loaded model
    prediction = model.predict(input_poly)
    
    # 3. Output result
    weight = max(0, prediction[0]) # Weight cannot be negative
    st.success(f"### Predicted Weight: {weight:.2f} grams")

    st.info(f"Model Accuracy (R²): 0.961")
