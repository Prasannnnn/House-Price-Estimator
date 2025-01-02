import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title of the Streamlit app
st.title("House Price Estimator")
st.write("Predict house prices based on size and number of rooms using machine learning.")

# Sidebar for dataset upload
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load dataset and preprocess
if uploaded_file:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(data.head())

    # Data Preprocessing
    data['Size(SqFt)'] = data['Size(SqFt)'].fillna(data['Size(SqFt)'].mean())
    data['Rooms'] = data['Rooms'].fillna(data['Rooms'].median())
    data = data.dropna(subset=['Price'])

    # Features and target
    X = data[['Size(SqFt)', 'Rooms']]
    y = data['Price']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"### Model Performance")
    st.write(f"Mean Squared Error (MSE): ₹{mse:,.2f}")

    # User input for prediction
    st.sidebar.header("Input Features for Prediction")
    size = st.sidebar.number_input("Enter Size (SqFt)", min_value=500, max_value=3000, value=1500, step=10)
    rooms = st.sidebar.number_input("Enter Number of Rooms", min_value=1, max_value=5, value=3, step=1)

    # Make a prediction based on user input
    if st.sidebar.button("Predict"):
        prediction = model.predict([[size, rooms]])[0]
        st.write(f"### Predicted House Price: ₹{prediction:,.2f}")
else:
    st.write("Please upload a dataset to proceed.")
