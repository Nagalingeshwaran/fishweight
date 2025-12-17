import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# App title
st.title("🐟 Fish Weight Prediction using Polynomial Regression")

st.write("""
This app predicts **Fish Weight** using **Polynomial Regression**  
based on fish body measurements.
""")

# Load dataset
df = pd.read_csv("Fish.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features and target
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# Degree selection
degree = st.slider("Select Polynomial Degree", min_value=1, max_value=4, value=2)

# Polynomial transformation
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegre
