import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('model.h5')

st.title('Demand Prediction App')

# User inputs for features without min and max constraints
year = st.number_input('Year', value=2024)
location = st.text_input('Location', value='London')  # Currently unused in the model
week = st.number_input('Week', value=1)  # Removed min_value and max_value
time = st.number_input('Time', value=12)  # Removed min_value and max_value

# Debug: Print input types to verify correctness
st.write(f"Year type: {type(year)}, Week type: {type(week)}, Time type: {type(time)}")

try:
    # Ensure inputs are cast to float32
    year = np.float32(year)
    week = np.float32(week)
    time = np.float32(time)

    # Prepare the input features array
    input_features = np.array([[year, week, time]], dtype=np.float32)

    # Reshape to match the model input shape: (batch_size, timesteps, features)
    input_features = input_features.reshape((1, 1, 3))  # Adjust shape based on model needs

    # Predict button
    if st.button('Predict Demand'):
        # Make prediction
        prediction = model.predict(input_features)
        st.write(f'Predicted Demand: {prediction[0][0]:.2f}')

except ValueError as e:
    st.error(f"Input error: {e}. Please check your inputs and ensure all are numbers.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")


