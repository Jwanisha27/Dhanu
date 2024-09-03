import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Title of the app
st.title('Your Model Prediction App')

# Description or instructions
st.write('This app predicts output using your machine learning model.')

# Load your model
# Replace 'model.h5' with the actual path to your saved model
model = load_model('model.h5')

# Define inputs
# Replace these inputs with those relevant to your model
year = st.number_input('Enter the Year', value=2024)
week = st.number_input('Enter the Week', value=1)  # Set sensible default
time = st.number_input('Enter the Time', value=12)

# Process inputs and predict
try:
    # Ensure inputs are correct type
    year = np.float32(year)
    week = np.float32(week)
    time = np.float32(time)
    
    # Prepare the input for the model
    input_features = np.array([[year, week, time]], dtype=np.float32)
    input_features = input_features.reshape((1, 1, 3))  # Adjust to model's expected shape

    # Button to trigger prediction
    if st.button('Predict'):
        prediction = model.predict(input_features)
        st.write(f'Predicted Value: {prediction[0][0]:.2f}')

except Exception as e:
    st.error(f"Error: {e}")
