import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('model.h5')

st.title('Demand Prediction App')

# User inputs for features
year = st.number_input('Year', value=2024)
location = st.text_input('Location', value='London')
week = st.number_input('Week', min_value=1, max_value=52, value=1)
time = st.number_input('Time', min_value=0, max_value=23, value=12)

# Example input processing (adjust as needed based on your feature setup)
# Convert input into the right shape for the model
# This is a placeholder; you might need to encode 'location' or adjust the array dimensions
input_features = np.array([[year, week, time]])  # Update this as necessary for your specific model

# Reshape to match model input shape
input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

# Predict button
if st.button('Predict Demand'):
    # Make prediction
    prediction = model.predict(input_features)
    st.write(f'Predicted Demand: {prediction[0][0]:.2f}')
