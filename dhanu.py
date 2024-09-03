import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Title of the app
st.title('Demand Prediction App')

# Description or instructions
st.write('This app predicts demand based on year, week, time, and location.')

# Load your model
# Replace 'model.h5' with the actual path to your saved model
model = load_model('model.h5')

# Define inputs
year = st.number_input('Enter the Year', value=2024)
week = st.number_input('Enter the Week', value=1)
time = st.number_input('Enter the Time', value=12)

# Location input as text
location = st.text_input('Enter Location', value='London')

# Encode location if needed (e.g., one-hot encoding, label encoding)
# For simplicity, here location input is just displayed but not used in prediction
# You would need to adapt this if your model uses location as input
st.write(f"Location: {location}")

try:
    # Ensure inputs are correct type
    year = np.float32(year)
    week = np.float32(week)
    time = np.float32(time)
    
    # Prepare the input for the model
    input_features = np.array([[year, week, time]], dtype=np.float32)
    input_features = input_features.reshape((1, 1, 3))  # Adjust to model's expected shape

    # Predict button
    if st.button('Predict Demand'):
        prediction = model.predict(input_features)
        st.write(f'Predicted Demand: {prediction[0][0]:.2f}')

except Exception as e:
    st.error(f"Error: {e}")
