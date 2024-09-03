import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('model.h5')

st.title('Demand Prediction App')

# User inputs for features without min and max constraints
year = st.number_input('Year', value=2024)
location = st.text_input('Location', value='London')  # This input is currently unused in the model
week = st.number_input('Week', value=1)  # Removed min_value and max_value
time = st.number_input('Time', value=12)  # Removed min_value and max_value

# Ensure that the inputs are numeric and cast them correctly
try:
    # Convert input into the correct format (numeric and reshaped)
    input_features = np.array([[year, week, time]], dtype=np.float32)  # Adjust to match your model's expected input

    # Reshape to match model input shape (batch_size, timesteps, features)
    input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))  # (1, 1, 3)

    # Predict button
    if st.button('Predict Demand'):
        # Make prediction
        prediction = model.predict(input_features)
        st.write(f'Predicted Demand: {prediction[0][0]:.2f}')
except ValueError as e:
    st.error(f"Input error: {e}. Please check your inputs and try again.")

