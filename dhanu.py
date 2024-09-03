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

try:
    # Convert input features to float32 and ensure the correct shape
    input_features = np.array([[float(year), float(week), float(time)]], dtype=np.float32)

    # Reshape input to match the model's expected input shape
    input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))  # (1, 1, 3)

    # Predict button
    if st.button('Predict Demand'):
        # Make prediction
        prediction = model.predict(input_features)
        st.write(f'Predicted Demand: {prediction[0][0]:.2f}')

except ValueError as e:
    st.error(f"Input error: {e}. Please check your inputs and ensure all are numbers.")


